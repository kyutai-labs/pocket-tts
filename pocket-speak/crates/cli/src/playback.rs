//! Plays samples as the engine produces them.
//!
//! The engine runs on the calling thread and hands pieces to a bounded
//! channel; the output device's callback drains it. The bound is what keeps
//! synthesis from racing minutes ahead of what has been heard, and blocking
//! there is safe because the engine loop is ours, not a callback that cannot
//! suspend (the Android reader's reason for a semaphore does not apply).
//! A closed receiver tells the producer to stop.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};

/// Pieces are ~1 s each (12 frames of 1920 samples), so this is a few seconds
/// of audio composed ahead of the listener.
const QUEUE_PIECES: usize = 4;

pub struct Player {
    sender: Option<SyncSender<Vec<f32>>>,
    resampler: Resampler,
    done: Arc<AtomicBool>,
    stream: cpal::Stream,
}

impl Player {
    pub fn open(source_rate: u32) -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or("no default audio output device")?;
        let supported = device
            .default_output_config()
            .map_err(|e| format!("audio output config: {e}"))?;
        let device_rate = supported.sample_rate();
        let channels = supported.channels() as usize;
        let format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();
        let (sender, receiver) = sync_channel(QUEUE_PIECES);
        let done = Arc::new(AtomicBool::new(false));
        let stream = match format {
            cpal::SampleFormat::F32 => {
                build::<f32>(&device, &config, channels, receiver, done.clone())
            }
            cpal::SampleFormat::I16 => {
                build::<i16>(&device, &config, channels, receiver, done.clone())
            }
            cpal::SampleFormat::U16 => {
                build::<u16>(&device, &config, channels, receiver, done.clone())
            }
            cpal::SampleFormat::I32 => {
                build::<i32>(&device, &config, channels, receiver, done.clone())
            }
            other => return Err(format!("unsupported output sample format {other:?}")),
        }?;
        stream.play().map_err(|e| format!("starting audio: {e}"))?;
        Ok(Self {
            sender: Some(sender),
            resampler: Resampler::new(source_rate, device_rate),
            done,
            stream,
        })
    }

    /// Queues samples at the source rate. Blocks while the queue is full;
    /// returns `false` if playback has gone away.
    pub fn push(&mut self, samples: &[f32]) -> bool {
        let piece = self.resampler.process(samples);
        match &self.sender {
            Some(sender) => sender.send(piece).is_ok(),
            None => false,
        }
    }

    pub fn push_silence(&mut self, seconds: f32, source_rate: u32) -> bool {
        let zeros = vec![0.0; (seconds * source_rate as f32) as usize];
        self.push(&zeros)
    }

    /// Waits until everything queued has been played.
    pub fn finish(mut self) {
        self.sender = None;
        while !self.done.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(20));
        }
        // The callback has handed its last samples to the device; give the
        // device buffer time to reach the speaker before the stream drops.
        std::thread::sleep(Duration::from_millis(250));
        drop(self.stream);
    }
}

fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    receiver: Receiver<Vec<f32>>,
    done: Arc<AtomicBool>,
) -> Result<cpal::Stream, String>
where
    T: SizedSample + FromSample<f32>,
{
    let mut current: Vec<f32> = Vec::new();
    let mut position = 0;
    device
        .build_output_stream(
            config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                for frame in data.chunks_mut(channels) {
                    if position >= current.len() {
                        match receiver.try_recv() {
                            Ok(next) => {
                                current = next;
                                position = 0;
                            }
                            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                                done.store(true, Ordering::Release);
                                current.clear();
                                position = 0;
                            }
                            Err(std::sync::mpsc::TryRecvError::Empty) => {
                                current.clear();
                                position = 0;
                            }
                        }
                    }
                    let sample = current.get(position).copied().unwrap_or(0.0);
                    position += usize::from(position < current.len());
                    for out in frame.iter_mut() {
                        *out = T::from_sample(sample);
                    }
                }
            },
            |error| eprintln!("pocket-speak: audio stream error: {error}"),
            None,
        )
        .map_err(|e| format!("opening audio output: {e}"))
}

/// Linear resampling, carried across calls so piece boundaries do not click.
/// Speech at 24 kHz to a 44.1 or 48 kHz device loses nothing linear
/// interpolation can hear.
pub struct Resampler {
    step: f64,
    position: f64,
    previous: f32,
}

impl Resampler {
    pub fn new(from: u32, to: u32) -> Self {
        Self {
            step: from as f64 / to as f64,
            position: 0.0,
            previous: 0.0,
        }
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if (self.step - 1.0).abs() < f64::EPSILON {
            return input.to_vec();
        }
        let mut output = Vec::with_capacity((input.len() as f64 / self.step) as usize + 2);
        // `position` counts from `previous`, which sits at index -1.
        while self.position < input.len() as f64 {
            let index = self.position.floor();
            let fraction = (self.position - index) as f32;
            let left = if index < 1.0 {
                self.previous
            } else {
                input[index as usize - 1]
            };
            let right = input[index as usize];
            output.push(left + (right - left) * fraction);
            self.position += self.step;
        }
        self.position -= input.len() as f64;
        if let Some(last) = input.last() {
            self.previous = *last;
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::Resampler;

    #[test]
    fn doubling_the_rate_doubles_the_length_across_calls() {
        let mut resampler = Resampler::new(24_000, 48_000);
        let first = resampler.process(&[0.0; 1000]);
        let second = resampler.process(&[0.0; 1000]);
        assert_eq!(first.len() + second.len(), 4000);
    }

    #[test]
    fn a_ramp_stays_a_ramp_over_a_piece_boundary() {
        let input: Vec<f32> = (0..200).map(|i| i as f32).collect();
        let mut resampler = Resampler::new(24_000, 48_000);
        let mut output = resampler.process(&input[..100]);
        output.extend(resampler.process(&input[100..]));
        for pair in output.windows(2).skip(2) {
            let step = pair[1] - pair[0];
            assert!((step - 0.5).abs() < 1e-3, "step {step} in {pair:?}");
        }
    }

    #[test]
    fn a_44_1_khz_device_gets_the_expected_sample_count() {
        let mut resampler = Resampler::new(24_000, 44_100);
        let total: usize = (0..24).map(|_| resampler.process(&[0.0; 1000]).len()).sum();
        assert!((total as i64 - 44_100).abs() <= 2, "{total}");
    }
}
