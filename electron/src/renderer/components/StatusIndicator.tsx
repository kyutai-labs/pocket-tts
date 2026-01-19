import React from 'react';
import type { GenerationStatus } from '../App';

interface StatusIndicatorProps {
  status: GenerationStatus;
  timeToFirstAudio: number | null;
  totalTime: number | null;
  error: string | null;
}

export function StatusIndicator({
  status,
  timeToFirstAudio,
  totalTime,
  error,
}: StatusIndicatorProps) {
  if (status === 'idle') return null;

  return (
    <div className={`rounded-lg p-4 ${status === 'error' ? 'bg-red-900/30' : 'bg-bg-secondary'}`}>
      <div className="flex items-center gap-3">
        {/* Status Icon */}
        {status === 'generating' && (
          <div className="flex items-center gap-2">
            <svg className="animate-spin h-4 w-4 text-accent" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <span className="text-sm text-text-primary">Generating speech...</span>
          </div>
        )}

        {status === 'streaming' && (
          <div className="flex items-center gap-2">
            <div className="flex gap-1">
              <div className="w-1.5 h-4 bg-accent rounded-full animate-pulse" style={{ animationDelay: '0ms' }} />
              <div className="w-1.5 h-4 bg-accent rounded-full animate-pulse" style={{ animationDelay: '150ms' }} />
              <div className="w-1.5 h-4 bg-accent rounded-full animate-pulse" style={{ animationDelay: '300ms' }} />
            </div>
            <span className="text-sm text-text-primary">
              Playing... (first audio in {timeToFirstAudio?.toFixed(2)}s)
            </span>
          </div>
        )}

        {status === 'complete' && (
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            <span className="text-sm text-text-primary">
              Complete! First audio: {timeToFirstAudio?.toFixed(2)}s | Total: {totalTime?.toFixed(2)}s
            </span>
          </div>
        )}

        {status === 'error' && (
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
            <span className="text-sm text-red-400">
              Error: {error}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
