import React from 'react';

interface TextInputProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

export function TextInput({ value, onChange, disabled }: TextInputProps) {
  const charCount = value.length;
  const wordCount = value.trim().split(/\s+/).filter(Boolean).length;

  return (
    <div className="bg-bg-secondary rounded-lg p-4">
      <label className="block text-sm font-medium text-text-primary mb-3">
        Text to Generate
      </label>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        placeholder="Enter the text you want to convert to speech..."
        rows={5}
        className={`w-full bg-bg-tertiary text-text-primary border border-border-color rounded-lg px-4 py-3 text-sm
          placeholder-text-secondary resize-y min-h-[120px]
          focus:outline-none focus:ring-2 focus:ring-accent focus:border-transparent
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      />
      <div className="mt-2 flex justify-between text-xs text-text-secondary">
        <span>{wordCount} words</span>
        <span>{charCount} characters</span>
      </div>
    </div>
  );
}
