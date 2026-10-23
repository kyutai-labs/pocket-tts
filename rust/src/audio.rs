//! Audio input and output: port of `pocket_tts/data/audio.py` and
//! `pocket_tts/data/audio_utils.py`.

use anyhow::{Context, Result};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

/// Reads an audio file as mono samples, averaging the channels, and its sample rate.
///
/// TODO: upstream to xn-ptts: `ptts::audio::decode_file` keeps only the first channel and does not
/// trim the MP3/AAC encoder delay, which libsndfile (the Python package's decoder) does.
pub fn audio_read(path: &Path) -> Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file =
        std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
        hint.with_extension(ext);
    }
    // Gapless: the encoder delay and padding of MP3 and AAC are trimmed, as libsndfile (the
    // Python package's decoder) does, rather than heard as a shift.
    let format_options = FormatOptions {
        enable_gapless: true,
        ..Default::default()
    };
    let probed = symphonia::default::get_probe()
        .format(&hint, stream, &format_options, &MetadataOptions::default())
        .with_context(|| format!("unsupported audio format: {}", path.display()))?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .with_context(|| format!("no audio track in {}", path.display()))?;
    let track_id = track.id;
    let mut sample_rate = track.codec_params.sample_rate;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .with_context(|| format!("unsupported audio codec in {}", path.display()))?;

    let mut samples = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(e).with_context(|| format!("cannot read {}", path.display())),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            // A corrupt packet is skipped, as decoders conventionally do.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(e).with_context(|| format!("cannot decode {}", path.display())),
        };
        let spec = *decoded.spec();
        sample_rate.get_or_insert(spec.rate);
        let channels = spec.channels.count();
        let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buffer.copy_interleaved_ref(decoded);
        for frame in buffer.samples().chunks_exact(channels) {
            samples.push(frame.iter().sum::<f32>() / channels as f32);
        }
    }
    let sample_rate =
        sample_rate.with_context(|| format!("no sample rate in {}", path.display()))?;
    Ok((samples, sample_rate))
}

/// End a voice prompt on exactly `pause_sec` of silence.
///
/// Training prompts end inside the pause between two words. A prompt that stops on speech makes
/// the model continue that utterance, and one that ends on a long silence delays the onset. The
/// trailing silence (20 ms frames more than `floor_db` below the loudest one) is cut, the last
/// `fade_sec` of what remains is faded out, and `pause_sec` of zeros is appended.
pub fn end_on_pause(wav: &[f32], sample_rate: usize) -> Vec<f32> {
    let (pause_sec, fade_sec, floor_db) = (0.08, 0.02, 35.0f32);
    let frame = ((0.02 * sample_rate as f64) as usize).max(1);
    let n = wav.len() / frame;
    if n == 0 {
        return wav.to_vec();
    }
    let db: Vec<f32> = wav[..n * frame]
        .chunks_exact(frame)
        .map(|chunk| {
            let rms = (chunk.iter().map(|v| v * v).sum::<f32>() / frame as f32).sqrt();
            20.0 * (rms + 1e-12).log10()
        })
        .collect();
    let max_db = db.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let last_loud = db.iter().rposition(|&d| d > max_db - floor_db).unwrap_or(0);
    let end = (last_loud + 1) * frame;
    let mut out = wav[..end].to_vec();
    let fade = ((fade_sec * sample_rate as f64) as usize).min(end);
    for i in 0..fade {
        let gain = if fade == 1 {
            1.0
        } else {
            1.0 - i as f32 / (fade - 1) as f32
        };
        out[end - fade + i] *= gain;
    }
    out.extend(std::iter::repeat_n(
        0.0,
        (pause_sec * sample_rate as f64) as usize,
    ));
    out
}

/// Seconds of audio buffered before the first write, from `FIRST_CHUNK_LENGTH_SECONDS`.
fn first_chunk_length_seconds() -> f64 {
    std::env::var("FIRST_CHUNK_LENGTH_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0)
}

/// An output a WAV is streamed to: a file, whose header is fixed up at the end, or a pipe.
pub trait WavSink: Write {
    /// Moves back to the start to rewrite the header; `false` when the sink cannot.
    fn rewind(&mut self) -> bool;
}

impl WavSink for std::fs::File {
    fn rewind(&mut self) -> bool {
        self.seek(SeekFrom::Start(0)).is_ok()
    }
}

/// A sink that cannot seek, such as a pipe or an HTTP response.
pub struct Unseekable<W: Write>(pub W);

impl<W: Write> Write for Unseekable<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

impl<W: Write> WavSink for Unseekable<W> {
    fn rewind(&mut self) -> bool {
        false
    }
}

/// Port of `StreamingWAVWriter`: 16-bit mono PCM, written as it is generated.
///
/// An unseekable output gets the header Python's `wave` module writes for it, announcing a
/// billion frames; a file gets its sizes fixed when the stream ends.
pub struct StreamingWavWriter<S: WavSink> {
    sink: S,
    sample_rate: u32,
    seekable: bool,
    data_bytes: u64,
    first_chunk_buffer: Option<Vec<u8>>,
}

const UNSEEKABLE_NFRAMES: u64 = 1_000_000_000;

impl<S: WavSink> StreamingWavWriter<S> {
    pub fn new(sink: S, sample_rate: u32, seekable: bool) -> Self {
        Self {
            sink,
            sample_rate,
            seekable,
            data_bytes: 0,
            first_chunk_buffer: Some(Vec::new()),
        }
    }

    fn write_header(&mut self, data_bytes: u64) -> std::io::Result<()> {
        let data_bytes = data_bytes.min(u32::MAX as u64 - 36) as u32;
        let mut header = Vec::with_capacity(44);
        header.extend_from_slice(b"RIFF");
        header.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        header.extend_from_slice(b"WAVEfmt ");
        header.extend_from_slice(&16u32.to_le_bytes());
        header.extend_from_slice(&1u16.to_le_bytes()); // PCM
        header.extend_from_slice(&1u16.to_le_bytes()); // mono
        header.extend_from_slice(&self.sample_rate.to_le_bytes());
        header.extend_from_slice(&(self.sample_rate * 2).to_le_bytes());
        header.extend_from_slice(&2u16.to_le_bytes());
        header.extend_from_slice(&16u16.to_le_bytes());
        header.extend_from_slice(b"data");
        header.extend_from_slice(&data_bytes.to_le_bytes());
        self.sink.write_all(&header)
    }

    pub fn write_header_initial(&mut self) -> std::io::Result<()> {
        let initial = if self.seekable {
            0
        } else {
            UNSEEKABLE_NFRAMES * 2
        };
        self.write_header(initial)
    }

    fn write_raw(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.data_bytes += bytes.len() as u64;
        self.sink.write_all(bytes)?;
        self.sink.flush()
    }

    pub fn write_pcm(&mut self, audio: &[f32]) -> std::io::Result<()> {
        let bytes: Vec<u8> = audio
            .iter()
            .flat_map(|&v| ((v.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes())
            .collect();
        if let Some(buffer) = self.first_chunk_buffer.as_mut() {
            buffer.extend_from_slice(&bytes);
            let target = (self.sample_rate as f64 * first_chunk_length_seconds()) as usize * 2;
            if buffer.len() < target {
                return Ok(());
            }
            return self.flush_first_chunk();
        }
        self.write_raw(&bytes)
    }

    fn flush_first_chunk(&mut self) -> std::io::Result<()> {
        if let Some(buffer) = self.first_chunk_buffer.take() {
            self.write_raw(&buffer)?;
        }
        Ok(())
    }

    /// Writes 200 ms of silence, to ensure proper playback, and fixes up the header.
    pub fn finalize(mut self) -> std::io::Result<S> {
        self.flush_first_chunk()?;
        let silence = vec![0u8; (self.sample_rate as f64 * 0.2) as usize * 2];
        self.write_raw(&silence)?;
        if self.seekable && self.sink.rewind() {
            self.write_header(self.data_bytes)?;
        }
        self.sink.flush()?;
        Ok(self.sink)
    }
}

/// Opens `path` for a WAV, `-` being stdout. Returns whether the output can be seeked, which a
/// stdout redirected to a file can.
pub fn open_output(path: &str) -> Result<(Box<dyn WavSinkBox>, bool)> {
    if path == "-" {
        let stdout = stdout_file()?;
        let seekable = stdout.is_some_and(|mut f| f.stream_position().is_ok());
        match stdout_file()? {
            Some(file) if seekable => Ok((Box::new(file), true)),
            _ => Ok((Box::new(Unseekable(std::io::stdout())), false)),
        }
    } else {
        let file = std::fs::File::create(path).with_context(|| format!("cannot create {path}"))?;
        Ok((Box::new(file), true))
    }
}

pub trait WavSinkBox: WavSink + Send {}
impl<T: WavSink + Send> WavSinkBox for T {}

impl WavSink for Box<dyn WavSinkBox> {
    fn rewind(&mut self) -> bool {
        (**self).rewind()
    }
}

/// A handle on stdout as a file, to seek it when it is redirected to one.
#[cfg(unix)]
fn stdout_file() -> Result<Option<std::fs::File>> {
    use std::os::fd::AsFd;
    let fd = std::io::stdout().as_fd().try_clone_to_owned()?;
    Ok(Some(std::fs::File::from(fd)))
}

#[cfg(windows)]
fn stdout_file() -> Result<Option<std::fs::File>> {
    use std::os::windows::io::AsHandle;
    let handle = std::io::stdout().as_handle().try_clone_to_owned()?;
    Ok(Some(std::fs::File::from(handle)))
}

#[cfg(not(any(unix, windows)))]
fn stdout_file() -> Result<Option<std::fs::File>> {
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn end_on_pause_trims_silence_and_appends_a_pause() {
        let sr = 1000;
        let mut wav = vec![0.5f32; 500];
        wav.extend(vec![0.0f32; 500]);
        let out = end_on_pause(&wav, sr);
        assert_eq!(out.len(), 500 + 80);
        assert_eq!(out[0], 0.5);
        assert_eq!(out[499], 0.0);
        assert!(out[500..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn unseekable_header_announces_a_billion_frames() {
        let mut out = Vec::new();
        {
            let mut writer = StreamingWavWriter::new(Unseekable(&mut out), 24000, false);
            writer.write_header_initial().unwrap();
            writer.write_pcm(&[0.5, -2.0]).unwrap();
            writer.finalize().unwrap();
        }
        assert_eq!(&out[40..44], &2_000_000_000u32.to_le_bytes());
        assert_eq!(&out[44..48], &[0xff, 0x3f, 0x01, 0x80]);
        assert_eq!(out.len(), 44 + 4 + 24000 / 5 * 2);
    }
}
