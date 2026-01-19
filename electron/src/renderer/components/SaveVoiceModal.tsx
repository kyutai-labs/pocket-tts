import React, { useState, useCallback } from 'react';
import { Modal } from './Modal';

interface SaveVoiceModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (name: string, description: string) => Promise<void>;
  fileName: string;
}

export function SaveVoiceModal({ isOpen, onClose, onSave, fileName }: SaveVoiceModalProps) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSave = useCallback(async () => {
    if (!name.trim()) {
      setError('Please enter a name for the voice');
      return;
    }

    setIsSaving(true);
    setError(null);

    try {
      await onSave(name.trim(), description.trim());
      setName('');
      setDescription('');
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save voice');
    } finally {
      setIsSaving(false);
    }
  }, [name, description, onSave, onClose]);

  const handleClose = useCallback(() => {
    setName('');
    setDescription('');
    setError(null);
    onClose();
  }, [onClose]);

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Save Voice Preset">
      <div className="space-y-4">
        <p className="text-sm text-text-secondary">
          Would you like to save this voice to your presets?
        </p>

        <div className="text-xs text-text-secondary bg-bg-tertiary rounded-lg px-3 py-2">
          Source: {fileName}
        </div>

        <div>
          <label className="block text-sm font-medium text-text-primary mb-2">
            Voice Name *
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g., My Voice, John's Voice"
            className="w-full bg-bg-tertiary text-text-primary border border-border-color rounded-lg px-4 py-2 text-sm
              focus:outline-none focus:ring-2 focus:ring-accent focus:border-transparent"
            autoFocus
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-text-primary mb-2">
            Description (optional)
          </label>
          <input
            type="text"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="e.g., Male, casual tone"
            className="w-full bg-bg-tertiary text-text-primary border border-border-color rounded-lg px-4 py-2 text-sm
              focus:outline-none focus:ring-2 focus:ring-accent focus:border-transparent"
          />
        </div>

        {error && (
          <div className="text-sm text-red-400 bg-red-400/10 rounded-lg px-3 py-2">
            {error}
          </div>
        )}

        <div className="flex gap-3 pt-2">
          <button
            onClick={handleClose}
            className="flex-1 px-4 py-2 text-sm text-text-primary bg-bg-tertiary hover:bg-border-color rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving || !name.trim()}
            className={`flex-1 px-4 py-2 text-sm text-white bg-accent hover:bg-accent-hover rounded-lg transition-colors
              ${isSaving || !name.trim() ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {isSaving ? 'Saving...' : 'Save Voice'}
          </button>
        </div>
      </div>
    </Modal>
  );
}
