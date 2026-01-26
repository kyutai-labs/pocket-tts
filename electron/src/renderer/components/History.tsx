import React, { useState, useEffect, useCallback } from 'react';

export interface HistoryEntry {
  id: string;
  type: 'single' | 'multi';
  timestamp: number;
  pinned: boolean;
  // Single voice data
  text?: string;
  voice?: string;
  voiceName?: string;
  // Multi-talk data
  script?: string;
  speakers?: {
    name: string;
    voice: string;
    voiceName?: string;
    customUrl?: string | null;
    fileData?: string | null;
    seed?: number | null;
  }[];
}

interface HistoryProps {
  onReuseSingle: (entry: HistoryEntry) => void;
  onReuseMulti: (entry: HistoryEntry) => void;
}

const STORAGE_KEY = 'pocket-tts-history';
const MAX_UNPINNED_ENTRIES = 30;

function loadHistory(): HistoryEntry[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function saveHistory(entries: HistoryEntry[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
}

export function addToHistory(entry: Omit<HistoryEntry, 'id' | 'timestamp' | 'pinned'>) {
  const history = loadHistory();

  const newEntry: HistoryEntry = {
    ...entry,
    id: `history-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    timestamp: Date.now(),
    pinned: false,
  };

  // Add new entry at the beginning
  history.unshift(newEntry);

  // Keep all pinned entries + last MAX_UNPINNED_ENTRIES unpinned
  const pinned = history.filter((e) => e.pinned);
  const unpinned = history.filter((e) => !e.pinned).slice(0, MAX_UNPINNED_ENTRIES);

  saveHistory([...pinned, ...unpinned].sort((a, b) => b.timestamp - a.timestamp));
}

export function History({ onReuseSingle, onReuseMulti }: HistoryProps) {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [filter, setFilter] = useState<'all' | 'single' | 'multi' | 'pinned'>('all');

  // Load history on mount
  useEffect(() => {
    setEntries(loadHistory());

    // Listen for storage changes (from other tabs/components)
    const handleStorage = () => setEntries(loadHistory());
    window.addEventListener('storage', handleStorage);

    // Also check periodically for updates from same window
    const interval = setInterval(() => setEntries(loadHistory()), 1000);

    return () => {
      window.removeEventListener('storage', handleStorage);
      clearInterval(interval);
    };
  }, []);

  const togglePin = useCallback((id: string) => {
    setEntries((prev) => {
      const updated = prev.map((e) =>
        e.id === id ? { ...e, pinned: !e.pinned } : e
      );
      saveHistory(updated);
      return updated;
    });
  }, []);

  const deleteEntry = useCallback((id: string) => {
    if (!window.confirm('Delete this history entry?')) return;

    setEntries((prev) => {
      const updated = prev.filter((e) => e.id !== id);
      saveHistory(updated);
      return updated;
    });
  }, []);

  const clearUnpinned = useCallback(() => {
    setEntries((prev) => {
      const updated = prev.filter((e) => e.pinned);
      saveHistory(updated);
      return updated;
    });
  }, []);

  const handleReuse = useCallback((entry: HistoryEntry) => {
    if (entry.type === 'single') {
      onReuseSingle(entry);
    } else {
      onReuseMulti(entry);
    }
  }, [onReuseSingle, onReuseMulti]);

  const filteredEntries = entries.filter((e) => {
    if (filter === 'all') return true;
    if (filter === 'pinned') return e.pinned;
    return e.type === filter;
  });

  const formatTimestamp = (ts: number) => {
    const date = new Date(ts);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  const truncateText = (text: string, maxLen: number = 80) => {
    if (text.length <= maxLen) return text;
    return text.slice(0, maxLen) + '...';
  };

  return (
    <div className="space-y-4">
      {/* Filter and Actions */}
      <div className="flex justify-between items-center">
        <div className="flex gap-2">
          {(['all', 'single', 'multi', 'pinned'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1 text-xs rounded-full transition-colors
                ${filter === f
                  ? 'bg-accent text-white'
                  : 'bg-bg-secondary text-text-secondary hover:bg-bg-tertiary'
                }`}
            >
              {f === 'all' ? 'All' : f === 'single' ? 'Single' : f === 'multi' ? 'Multi-Talk' : 'Pinned'}
            </button>
          ))}
        </div>

        {entries.some((e) => !e.pinned) && (
          <button
            onClick={clearUnpinned}
            className="text-xs text-red-400 hover:text-red-300 transition-colors"
          >
            Clear Unpinned
          </button>
        )}
      </div>

      {/* History List */}
      {filteredEntries.length === 0 ? (
        <div className="text-center py-12 text-text-secondary">
          <p className="text-sm">No history yet</p>
          <p className="text-xs mt-1">Generated audio will appear here</p>
        </div>
      ) : (
        <div className="space-y-2">
          {filteredEntries.map((entry) => (
            <div
              key={entry.id}
              className={`bg-bg-secondary rounded-lg p-4 border transition-colors
                ${entry.pinned ? 'border-accent/50' : 'border-border-color'}`}
            >
              <div className="flex justify-between items-start mb-2">
                <div className="flex items-center gap-2">
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full
                      ${entry.type === 'single'
                        ? 'bg-blue-500/20 text-blue-400'
                        : 'bg-purple-500/20 text-purple-400'
                      }`}
                  >
                    {entry.type === 'single' ? 'Single' : 'Multi-Talk'}
                  </span>
                  <span className="text-xs text-text-secondary">
                    {formatTimestamp(entry.timestamp)}
                  </span>
                  {entry.pinned && (
                    <span className="text-xs text-accent">Pinned</span>
                  )}
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => togglePin(entry.id)}
                    className={`text-sm transition-colors
                      ${entry.pinned
                        ? 'text-accent hover:text-accent-hover'
                        : 'text-text-secondary hover:text-text-primary'
                      }`}
                    title={entry.pinned ? 'Unpin' : 'Pin'}
                  >
                    {entry.pinned ? '★' : '☆'}
                  </button>
                  <button
                    onClick={() => deleteEntry(entry.id)}
                    className="text-sm text-text-secondary hover:text-red-400 transition-colors"
                    title="Delete"
                  >
                    ×
                  </button>
                </div>
              </div>

              {/* Entry Content */}
              <div className="text-sm text-text-primary mb-3">
                {entry.type === 'single' ? (
                  <>
                    <p className="text-xs text-text-secondary mb-1">
                      Voice: {entry.voiceName || entry.voice}
                    </p>
                    <p className="font-mono text-xs bg-bg-tertiary rounded p-2">
                      {truncateText(entry.text || '')}
                    </p>
                  </>
                ) : (
                  <>
                    <p className="text-xs text-text-secondary mb-1">
                      Speakers: {entry.speakers?.map((s) => s.name).join(', ')}
                    </p>
                    <p className="font-mono text-xs bg-bg-tertiary rounded p-2">
                      {truncateText(entry.script || '')}
                    </p>
                  </>
                )}
              </div>

              {/* Reuse Button */}
              <button
                onClick={() => handleReuse(entry)}
                className="w-full py-2 text-sm bg-bg-tertiary hover:bg-accent/20
                  text-text-secondary hover:text-accent rounded transition-colors"
              >
                Reuse Setup
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
