import React, { useRef, useState, useEffect, useCallback } from 'react';

interface AudioPlayerProps {
  audioBlob: Blob;
}

export function AudioPlayer({ audioBlob }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  useEffect(() => {
    console.log('[AudioPlayer] Received audio blob:', {
      size: audioBlob.size,
      type: audioBlob.type,
    });
    const url = URL.createObjectURL(audioBlob);
    console.log('[AudioPlayer] Created blob URL:', url);
    setAudioUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [audioBlob]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !audioUrl) return;

    const handleTimeUpdate = () => setCurrentTime(audio.currentTime);
    const handleLoadedMetadata = () => {
      console.log('[AudioPlayer] loadedmetadata event - duration:', audio.duration);
      if (isFinite(audio.duration)) {
        setDuration(audio.duration);
      }
    };
    const handleDurationChange = () => {
      console.log('[AudioPlayer] durationchange event - duration:', audio.duration);
      if (isFinite(audio.duration)) {
        setDuration(audio.duration);
      }
    };
    const handleCanPlayThrough = () => {
      console.log('[AudioPlayer] canplaythrough event - duration:', audio.duration);
      if (isFinite(audio.duration)) {
        setDuration(audio.duration);
      }
    };
    const handleEnded = () => setIsPlaying(false);
    const handlePlay = () => {
      console.log('[AudioPlayer] play event fired');
      setIsPlaying(true);
    };
    const handlePause = () => setIsPlaying(false);
    const handleError = (e: Event) => {
      console.error('[AudioPlayer] Audio error:', (e.target as HTMLAudioElement).error);
    };

    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('durationchange', handleDurationChange);
    audio.addEventListener('canplaythrough', handleCanPlayThrough);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);
    audio.addEventListener('error', handleError);

    // Force load metadata
    console.log('[AudioPlayer] Calling audio.load()');
    audio.load();

    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('durationchange', handleDurationChange);
      audio.removeEventListener('canplaythrough', handleCanPlayThrough);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
      audio.removeEventListener('error', handleError);
    };
  }, [audioUrl]);

  const togglePlayPause = useCallback(async () => {
    const audio = audioRef.current;
    console.log('[AudioPlayer] Play button clicked, audio element:', audio);
    console.log('[AudioPlayer] Current state:', { isPlaying, currentTime, duration, audioUrl });
    if (!audio) {
      console.error('[AudioPlayer] No audio element ref!');
      return;
    }

    if (isPlaying) {
      console.log('[AudioPlayer] Pausing...');
      audio.pause();
    } else {
      console.log('[AudioPlayer] Playing... readyState:', audio.readyState, 'src:', audio.src);
      try {
        await audio.play();
        console.log('[AudioPlayer] Play started successfully');
      } catch (error) {
        console.error('[AudioPlayer] Failed to play audio:', error);
      }
    }
  }, [isPlaying, currentTime, duration, audioUrl]);

  const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    if (!audio) return;

    const time = parseFloat(e.target.value);
    audio.currentTime = time;
    setCurrentTime(time);
  }, []);

  const handleDownload = useCallback(() => {
    if (!audioUrl) return;

    const a = document.createElement('a');
    a.href = audioUrl;
    a.download = 'pocket-tts-output.wav';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [audioUrl]);

  const formatTime = (time: number) => {
    if (!isFinite(time)) return '0:00';
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  if (!audioUrl) return null;

  return (
    <div className="bg-bg-secondary rounded-lg p-4">
      <audio ref={audioRef} src={audioUrl} preload="metadata" />

      <div className="flex items-center gap-4">
        {/* Play/Pause Button */}
        <button
          onClick={togglePlayPause}
          className="w-12 h-12 flex items-center justify-center rounded-full bg-accent hover:bg-accent-hover transition-colors"
        >
          {isPlaying ? (
            <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
              <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
            </svg>
          ) : (
            <svg className="w-5 h-5 text-white ml-1" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
        </button>

        {/* Progress Bar */}
        <div className="flex-1">
          <input
            type="range"
            min={0}
            max={duration || 0}
            value={currentTime}
            onChange={handleSeek}
            className="w-full h-2 bg-bg-tertiary rounded-lg appearance-none cursor-pointer
              [&::-webkit-slider-thumb]:appearance-none
              [&::-webkit-slider-thumb]:w-3
              [&::-webkit-slider-thumb]:h-3
              [&::-webkit-slider-thumb]:rounded-full
              [&::-webkit-slider-thumb]:bg-accent"
          />
          <div className="flex justify-between text-xs text-text-secondary mt-1">
            <span>{formatTime(currentTime)}</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>

        {/* Download Button */}
        <button
          onClick={handleDownload}
          className="w-10 h-10 flex items-center justify-center rounded-lg bg-bg-tertiary hover:bg-border-color transition-colors"
          title="Download audio"
        >
          <svg className="w-5 h-5 text-text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
        </button>
      </div>
    </div>
  );
}
