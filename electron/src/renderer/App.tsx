import React, { useState, useCallback, useRef, useEffect } from 'react';
import { ReferenceAudio } from './components/ReferenceAudio';
import { VoiceSelector, SavedVoice, PREDEFINED_VOICES } from './components/VoiceSelector';
import { TextInput } from './components/TextInput';
import { SynthesizeButton } from './components/SynthesizeButton';
import { AudioPlayer } from './components/AudioPlayer';
import { StatusIndicator } from './components/StatusIndicator';
import { SaveVoiceModal } from './components/SaveVoiceModal';
import { MultiTalk, MultiTalkConfig } from './components/MultiTalk';
import { History, HistoryEntry, addToHistory } from './components/History';
import { StreamingWavPlayer } from './lib/streaming-wav-player';
import './types/electron.d.ts';

export type GenerationStatus = 'idle' | 'generating' | 'streaming' | 'complete' | 'error';

type TabType = 'single' | 'multi' | 'history';

interface GenerationState {
  status: GenerationStatus;
  timeToFirstAudio: number | null;
  totalTime: number | null;
  error: string | null;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<TabType>('single');
  const [text, setText] = useState(
    "Hello world. I am Kyutai's Pocket TTS. I'm fast enough to run on small CPUs. I hope you'll like me."
  );
  const [selectedVoice, setSelectedVoice] = useState('alba');
  const [customAudioFile, setCustomAudioFile] = useState<File | null>(null);
  const [generationState, setGenerationState] = useState<GenerationState>({
    status: 'idle',
    timeToFirstAudio: null,
    totalTime: null,
    error: null,
  });
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [savedVoices, setSavedVoices] = useState<SavedVoice[]>([]);
  const [showSaveVoiceModal, setShowSaveVoiceModal] = useState(false);

  const playerRef = useRef<StreamingWavPlayer | null>(null);
  const startTimeRef = useRef<number>(0);
  const [pendingMultiConfig, setPendingMultiConfig] = useState<MultiTalkConfig | null>(null);

  // Load saved voices on startup
  useEffect(() => {
    window.electronAPI?.getSavedVoices().then(setSavedVoices);
  }, []);

  useEffect(() => {
    return () => {
      window.electronAPI?.removeAllListeners();
      playerRef.current?.stop();
    };
  }, []);

  const handleGenerate = useCallback(async () => {
    if (!text.trim()) return;

    // Reset state
    setGenerationState({
      status: 'generating',
      timeToFirstAudio: null,
      totalTime: null,
      error: null,
    });
    setAudioBlob(null);
    playerRef.current?.stop();

    startTimeRef.current = performance.now();

    // Set up streaming player
    playerRef.current = new StreamingWavPlayer({
      onFirstAudio: () => {
        const timeToFirst = (performance.now() - startTimeRef.current) / 1000;
        setGenerationState((prev) => ({
          ...prev,
          status: 'streaming',
          timeToFirstAudio: timeToFirst,
        }));
      },
      onComplete: () => {
        const totalTime = (performance.now() - startTimeRef.current) / 1000;
        setGenerationState((prev) => ({
          ...prev,
          status: 'complete',
          totalTime,
        }));
        if (playerRef.current) {
          setAudioBlob(playerRef.current.getAudioBlob());
        }

        // Save to history
        let voiceName = selectedVoice;
        if (selectedVoice.startsWith('saved:')) {
          const saved = savedVoices.find((v) => `saved:${v.id}` === selectedVoice);
          if (saved) voiceName = saved.name;
        } else if (selectedVoice !== 'custom') {
          const predefined = PREDEFINED_VOICES.find((v) => v.id === selectedVoice);
          if (predefined) voiceName = predefined.name;
        }
        addToHistory({
          type: 'single',
          text,
          voice: selectedVoice,
          voiceName,
        });
      },
      onError: (error) => {
        setGenerationState((prev) => ({
          ...prev,
          status: 'error',
          error: error.message,
        }));
      },
    });

    // Set up IPC listeners
    window.electronAPI.removeAllListeners();

    window.electronAPI.onTTSChunk((chunk) => {
      playerRef.current?.addChunk(new Uint8Array(chunk));
    });

    window.electronAPI.onTTSComplete(() => {
      playerRef.current?.flushRemaining();
    });

    window.electronAPI.onTTSError((error) => {
      setGenerationState((prev) => ({
        ...prev,
        status: 'error',
        error,
      }));
    });

    // Prepare TTS parameters
    let voiceFile: ArrayBuffer | undefined;
    let voiceUrl: string | undefined;
    let savedVoiceId: string | undefined;

    if (customAudioFile) {
      voiceFile = await customAudioFile.arrayBuffer();
    } else if (selectedVoice.startsWith('saved:')) {
      savedVoiceId = selectedVoice.replace('saved:', '');
    } else if (selectedVoice !== 'custom') {
      voiceUrl = selectedVoice;
    }

    // Start generation
    try {
      await window.electronAPI.generateTTS({
        text,
        voiceUrl,
        voiceFile,
        savedVoiceId,
      });
    } catch (error) {
      setGenerationState((prev) => ({
        ...prev,
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
      }));
    }
  }, [text, selectedVoice, customAudioFile, savedVoices]);

  const handleVoiceChange = useCallback((voice: string) => {
    setSelectedVoice(voice);
    if (voice !== 'custom' && !voice.startsWith('saved:')) {
      setCustomAudioFile(null);
    }
  }, []);

  const handleCustomAudio = useCallback((file: File | null) => {
    setCustomAudioFile(file);
    if (file) {
      setSelectedVoice('custom');
      // Show modal to optionally save the voice
      setShowSaveVoiceModal(true);
    }
  }, []);

  const handleSaveVoice = useCallback(async (name: string, description: string) => {
    if (!customAudioFile) return;

    const audioData = await customAudioFile.arrayBuffer();
    const savedVoice = await window.electronAPI.saveVoice({
      name,
      description,
      audioData,
    });

    setSavedVoices((prev) => [...prev, savedVoice]);
    setSelectedVoice(`saved:${savedVoice.id}`);
    setCustomAudioFile(null);
  }, [customAudioFile]);

  const handleDeleteSavedVoice = useCallback(async (id: string) => {
    await window.electronAPI.deleteVoice(id);
    setSavedVoices((prev) => prev.filter((v) => v.id !== id));
    setSelectedVoice('alba');
  }, []);

  // History reuse handlers
  const handleReuseSingle = useCallback((entry: HistoryEntry) => {
    if (entry.text) setText(entry.text);
    if (entry.voice) setSelectedVoice(entry.voice);
    setCustomAudioFile(null);
    setActiveTab('single');
  }, []);

  const handleReuseMulti = useCallback((entry: HistoryEntry) => {
    if (entry.script && entry.speakers) {
      setPendingMultiConfig({
        script: entry.script,
        speakers: entry.speakers,
      });
    }
    setActiveTab('multi');
  }, []);

  const handleMultiConfigLoaded = useCallback(() => {
    setPendingMultiConfig(null);
  }, []);

  const isGenerating = generationState.status === 'generating' || generationState.status === 'streaming';

  return (
    <div className="min-h-screen bg-bg-primary">
      {/* Drag region for macOS with dev tools toggle */}
      <div className="h-8 drag-region relative">
        <button
          onClick={() => window.electronAPI?.toggleDevTools()}
          className="absolute right-2 top-1 no-drag text-text-secondary/40 hover:text-text-secondary text-xs px-1.5 py-0.5 rounded transition-colors"
          title="Toggle Developer Tools"
        >
          DEV
        </button>
      </div>

      <div className="max-w-2xl mx-auto px-6 pb-8">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold text-text-primary">Pocket TTS</h1>
          <p className="text-sm text-text-secondary mt-1">
            High-quality text-to-speech that runs on your CPU
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b border-border-color mb-6">
          <button
            onClick={() => setActiveTab('single')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors
              ${activeTab === 'single'
                ? 'border-accent text-accent'
                : 'border-transparent text-text-secondary hover:text-text-primary'
              }`}
          >
            Single Voice
          </button>
          <button
            onClick={() => setActiveTab('multi')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors
              ${activeTab === 'multi'
                ? 'border-accent text-accent'
                : 'border-transparent text-text-secondary hover:text-text-primary'
              }`}
          >
            Multi-Talk
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors
              ${activeTab === 'history'
                ? 'border-accent text-accent'
                : 'border-transparent text-text-secondary hover:text-text-primary'
              }`}
          >
            History
          </button>
        </div>

        {/* Single Voice Tab */}
        {activeTab === 'single' && (
          <>
        {/* Reference Audio Section */}
        <div className="mb-6">
          <ReferenceAudio
            onFileSelect={handleCustomAudio}
            selectedFile={customAudioFile}
            disabled={isGenerating}
          />
        </div>

        {/* Voice Selector */}
        <div className="mb-6">
          <VoiceSelector
            selectedVoice={selectedVoice}
            onVoiceChange={handleVoiceChange}
            hasCustomAudio={!!customAudioFile}
            disabled={isGenerating}
            savedVoices={savedVoices}
            onDeleteSavedVoice={handleDeleteSavedVoice}
          />
        </div>

        {/* Text Input */}
        <div className="mb-6">
          <TextInput
            value={text}
            onChange={setText}
            disabled={isGenerating}
          />
        </div>

        {/* Synthesize Button */}
        <div className="mb-6">
          <SynthesizeButton
            onClick={handleGenerate}
            isGenerating={isGenerating}
            disabled={!text.trim()}
          />
        </div>

        {/* Status Indicator */}
        <StatusIndicator
          status={generationState.status}
          timeToFirstAudio={generationState.timeToFirstAudio}
          totalTime={generationState.totalTime}
          error={generationState.error}
        />

        {/* Audio Player */}
        {audioBlob && (
          <div className="mt-6">
            <AudioPlayer audioBlob={audioBlob} />
          </div>
        )}
          </>
        )}

        {/* Multi-Talk Tab */}
        {activeTab === 'multi' && (
          <MultiTalk
            pendingConfig={pendingMultiConfig}
            onConfigLoaded={handleMultiConfigLoaded}
          />
        )}

        {/* History Tab */}
        {activeTab === 'history' && (
          <History
            onReuseSingle={handleReuseSingle}
            onReuseMulti={handleReuseMulti}
          />
        )}
      </div>

      {/* Save Voice Modal */}
      <SaveVoiceModal
        isOpen={showSaveVoiceModal}
        onClose={() => setShowSaveVoiceModal(false)}
        onSave={handleSaveVoice}
        fileName={customAudioFile?.name ?? 'Unknown'}
      />
    </div>
  );
}
