import React, { useState, useCallback, useRef } from 'react';
import { ensureWavFormat, convertToWav } from '../lib/audio-utils';

interface ReferenceAudioProps {
  onFileSelect: (file: File | null) => void;
  selectedFile: File | null;
  disabled?: boolean;
}

export function ReferenceAudio({ onFileSelect, selectedFile, disabled }: ReferenceAudioProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isConverting, setIsConverting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) {
      setIsDragging(true);
    }
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled || isConverting) return;

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('audio/') || file.name.endsWith('.wav') || file.name.endsWith('.mp3')) {
        setIsConverting(true);
        try {
          const wavFile = await ensureWavFormat(file);
          onFileSelect(wavFile);
        } catch (error) {
          console.error('Failed to convert audio:', error);
        } finally {
          setIsConverting(false);
        }
      }
    }
  }, [disabled, isConverting, onFileSelect]);

  const handleFileInput = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0 && !isConverting) {
      setIsConverting(true);
      try {
        const wavFile = await ensureWavFormat(files[0]);
        onFileSelect(wavFile);
      } catch (error) {
        console.error('Failed to convert audio:', error);
      } finally {
        setIsConverting(false);
      }
    }
  }, [isConverting, onFileSelect]);

  const handleClick = useCallback(() => {
    if (!disabled) {
      fileInputRef.current?.click();
    }
  }, [disabled]);

  const startRecording = useCallback(async () => {
    if (disabled) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());

        // MediaRecorder outputs WebM, not WAV - convert it
        const recordedBlob = new Blob(chunksRef.current, { type: mediaRecorder.mimeType });
        setIsConverting(true);
        try {
          const wavBlob = await convertToWav(recordedBlob);
          const file = new File([wavBlob], 'recorded-voice.wav', { type: 'audio/wav' });
          onFileSelect(file);
        } catch (error) {
          console.error('Failed to convert recording:', error);
        } finally {
          setIsConverting(false);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Failed to start recording:', error);
    }
  }, [disabled, onFileSelect]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  const clearSelection = useCallback(() => {
    onFileSelect(null);
  }, [onFileSelect]);

  return (
    <div className="bg-bg-secondary rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <label className="text-sm font-medium text-text-primary flex items-center gap-2">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
          </svg>
          Reference Audio
        </label>
        {selectedFile && (
          <button
            onClick={clearSelection}
            className="text-xs text-text-secondary hover:text-text-primary"
            disabled={disabled}
          >
            Clear
          </button>
        )}
      </div>

      {isConverting ? (
        <div className="flex items-center justify-center gap-3 p-6 bg-bg-tertiary rounded-lg">
          <svg className="w-6 h-6 text-accent animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          <span className="text-sm text-text-secondary">Converting to WAV...</span>
        </div>
      ) : selectedFile ? (
        <div className="flex items-center gap-3 p-3 bg-bg-tertiary rounded-lg">
          <svg className="w-8 h-8 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2z" />
          </svg>
          <div className="flex-1 min-w-0">
            <p className="text-sm text-text-primary truncate">{selectedFile.name}</p>
            <p className="text-xs text-text-secondary">
              {(selectedFile.size / 1024).toFixed(1)} KB
            </p>
          </div>
        </div>
      ) : (
        <div
          className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors
            ${isDragging ? 'border-accent bg-accent/10' : 'border-border-color hover:border-text-secondary'}
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={handleClick}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*,.wav,.mp3"
            onChange={handleFileInput}
            className="hidden"
            disabled={disabled}
          />

          <svg className="w-10 h-10 mx-auto mb-3 text-text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>

          <p className="text-sm text-text-primary mb-1">Drop Audio Here</p>
          <p className="text-xs text-text-secondary">- or -</p>
          <p className="text-sm text-text-primary">Click to Upload</p>
        </div>
      )}

      {/* Microphone button */}
      <div className="mt-3 flex justify-center">
        <button
          onClick={isRecording ? stopRecording : startRecording}
          disabled={disabled || isConverting}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-colors
            ${isRecording
              ? 'bg-red-600 hover:bg-red-700 text-white animate-pulse-recording'
              : 'bg-bg-tertiary hover:bg-border-color text-text-primary'}
            ${disabled || isConverting ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
          </svg>
          {isRecording ? 'Stop Recording' : 'Record Voice'}
        </button>
      </div>
    </div>
  );
}
