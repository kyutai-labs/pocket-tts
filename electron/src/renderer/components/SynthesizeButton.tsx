import React from 'react';

interface SynthesizeButtonProps {
  onClick: () => void;
  isGenerating: boolean;
  disabled?: boolean;
}

export function SynthesizeButton({ onClick, isGenerating, disabled }: SynthesizeButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled || isGenerating}
      className={`w-full py-4 rounded-lg text-white font-semibold text-lg transition-colors
        ${disabled || isGenerating
          ? 'bg-gray-600 cursor-not-allowed'
          : 'bg-accent hover:bg-accent-hover active:bg-accent'
        }`}
    >
      {isGenerating ? (
        <span className="flex items-center justify-center gap-3">
          <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          Synthesizing...
        </span>
      ) : (
        'Synthesize'
      )}
    </button>
  );
}
