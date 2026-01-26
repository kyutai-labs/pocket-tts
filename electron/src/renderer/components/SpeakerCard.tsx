import React, { useRef } from 'react';
import { PREDEFINED_VOICES, SavedVoice } from './VoiceSelector';

export interface Speaker {
  id: string;
  name: string;
  voice: string;
  seed: number | null;
  customUrl: string | null;
  fileData: string | null;
  fileName: string | null;
}

interface SpeakerCardProps {
  speaker: Speaker;
  onUpdate: (updates: Partial<Speaker>) => void;
  onRemove: () => void;
  onInsertToScript: (name: string) => void;
  canRemove: boolean;
  disabled?: boolean;
  savedVoices?: SavedVoice[];
}

export function SpeakerCard({
  speaker,
  onUpdate,
  onRemove,
  onInsertToScript,
  canRemove,
  disabled,
  savedVoices = [],
}: SpeakerCardProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        const base64 = (reader.result as string).split(',')[1];
        onUpdate({ fileData: base64, fileName: file.name, voice: 'upload' });
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="bg-bg-tertiary rounded-lg p-4 border border-border-color">
      <div className="flex justify-between items-start mb-3">
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={speaker.name}
            onChange={(e) => onUpdate({ name: e.target.value })}
            placeholder="Speaker Name"
            disabled={disabled}
            className={`bg-transparent text-text-primary font-medium border-b border-transparent
              hover:border-border-color focus:border-accent focus:outline-none
              ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          />
          <button
            onClick={() => onInsertToScript(speaker.name)}
            disabled={disabled || !speaker.name.trim()}
            title={`Insert {${speaker.name}} into script`}
            className={`w-5 h-5 rounded-full bg-green-500 flex items-center justify-center
              hover:bg-green-400 transition-colors flex-shrink-0
              ${disabled || !speaker.name.trim() ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <span className="text-white text-xs font-bold leading-none">+</span>
          </button>
        </div>
        {canRemove && (
          <button
            onClick={onRemove}
            disabled={disabled}
            className={`text-red-400 text-sm hover:text-red-300 transition-colors
              ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            Remove
          </button>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-text-secondary mb-1">Voice</label>
          <select
            value={speaker.voice}
            onChange={(e) => onUpdate({ voice: e.target.value })}
            disabled={disabled}
            className={`w-full bg-bg-secondary text-text-primary border border-border-color rounded px-2 py-1.5 text-sm
              focus:outline-none focus:ring-1 focus:ring-accent
              ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
          >
            <optgroup label="Built-in Voices">
              {PREDEFINED_VOICES.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.name} - {v.description}
                </option>
              ))}
            </optgroup>
            {savedVoices.length > 0 && (
              <optgroup label="My Saved Voices">
                {savedVoices.map((v) => (
                  <option key={`saved:${v.id}`} value={`saved:${v.id}`}>
                    {v.name}{v.description ? ` - ${v.description}` : ''}
                  </option>
                ))}
              </optgroup>
            )}
            <optgroup label="Custom">
              <option value="custom_url">Custom URL...</option>
              <option value="upload">Upload WAV...</option>
            </optgroup>
          </select>
        </div>

        <div>
          <label className="block text-xs text-text-secondary mb-1">Seed (optional)</label>
          <input
            type="number"
            value={speaker.seed ?? ''}
            onChange={(e) => onUpdate({ seed: e.target.value ? parseInt(e.target.value) : null })}
            placeholder="Random"
            disabled={disabled}
            className={`w-full bg-bg-secondary text-text-primary border border-border-color rounded px-2 py-1.5 text-sm
              focus:outline-none focus:ring-1 focus:ring-accent
              ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          />
        </div>
      </div>

      {speaker.voice === 'custom_url' && (
        <div className="mt-3">
          <input
            type="text"
            value={speaker.customUrl ?? ''}
            onChange={(e) => onUpdate({ customUrl: e.target.value })}
            placeholder="hf://... or https://..."
            disabled={disabled}
            className={`w-full bg-bg-secondary text-text-primary border border-border-color rounded px-2 py-1.5 text-sm
              focus:outline-none focus:ring-1 focus:ring-accent
              ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          />
        </div>
      )}

      {speaker.voice === 'upload' && (
        <div className="mt-3">
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,audio/wav"
            onChange={handleFileChange}
            disabled={disabled}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled}
            className={`w-full bg-bg-secondary text-text-primary border border-border-color rounded px-2 py-1.5 text-sm
              hover:bg-bg-primary transition-colors
              ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {speaker.fileName ? speaker.fileName : 'Choose WAV file...'}
          </button>
          {speaker.fileData && (
            <span className="text-xs text-green-400 mt-1 block">File loaded</span>
          )}
        </div>
      )}
    </div>
  );
}
