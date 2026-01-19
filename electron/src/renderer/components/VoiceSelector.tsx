import React from 'react';

export const PREDEFINED_VOICES = [
  { id: 'alba', name: 'Alba', description: 'Female, casual' },
  { id: 'marius', name: 'Marius', description: 'Male' },
  { id: 'javert', name: 'Javert', description: 'Male' },
  { id: 'jean', name: 'Jean', description: 'Male' },
  { id: 'fantine', name: 'Fantine', description: 'Female' },
  { id: 'cosette', name: 'Cosette', description: 'Female, expressive' },
  { id: 'eponine', name: 'Eponine', description: 'Female' },
  { id: 'azelma', name: 'Azelma', description: 'Female' },
];

export interface SavedVoice {
  id: string;
  name: string;
  description: string;
  filePath: string;
  createdAt: string;
}

interface VoiceSelectorProps {
  selectedVoice: string;
  onVoiceChange: (voice: string) => void;
  hasCustomAudio?: boolean;
  disabled?: boolean;
  savedVoices?: SavedVoice[];
  onDeleteSavedVoice?: (id: string) => void;
}

export function VoiceSelector({
  selectedVoice,
  onVoiceChange,
  hasCustomAudio,
  disabled,
  savedVoices = [],
  onDeleteSavedVoice,
}: VoiceSelectorProps) {
  return (
    <div className="bg-bg-secondary rounded-lg p-4">
      <label className="block text-sm font-medium text-text-primary mb-3">
        Voice
      </label>
      <select
        value={selectedVoice}
        onChange={(e) => onVoiceChange(e.target.value)}
        disabled={disabled}
        className={`w-full bg-bg-tertiary text-text-primary border border-border-color rounded-lg px-4 py-3 text-sm
          focus:outline-none focus:ring-2 focus:ring-accent focus:border-transparent
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
      >
        <optgroup label="Built-in Voices">
          {PREDEFINED_VOICES.map((voice) => (
            <option key={voice.id} value={voice.id}>
              {voice.name} - {voice.description}
            </option>
          ))}
        </optgroup>
        {savedVoices.length > 0 && (
          <optgroup label="My Saved Voices">
            {savedVoices.map((voice) => (
              <option key={`saved:${voice.id}`} value={`saved:${voice.id}`}>
                {voice.name}{voice.description ? ` - ${voice.description}` : ''}
              </option>
            ))}
          </optgroup>
        )}
        {hasCustomAudio && (
          <optgroup label="Current Session">
            <option value="custom">Custom (uploaded audio)</option>
          </optgroup>
        )}
      </select>
      <p className="mt-2 text-xs text-text-secondary">
        Select a pre-made voice or upload custom audio above for voice cloning.
      </p>

      {/* Show delete button for saved voices */}
      {selectedVoice.startsWith('saved:') && onDeleteSavedVoice && (
        <button
          onClick={() => onDeleteSavedVoice(selectedVoice.replace('saved:', ''))}
          disabled={disabled}
          className="mt-2 text-xs text-red-400 hover:text-red-300 transition-colors"
        >
          Delete this saved voice
        </button>
      )}
    </div>
  );
}
