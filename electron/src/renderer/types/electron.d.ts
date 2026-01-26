export interface TTSParams {
  text: string;
  voiceUrl?: string;
  voiceFile?: ArrayBuffer;
  savedVoiceId?: string;
}

export interface SavedVoice {
  id: string;
  name: string;
  description: string;
  filePath: string;
  createdAt: string;
}

export interface SpeakerConfig {
  name: string;
  voice_source: string;
  voice_data: string | null;
  seed: number | null;
}

export interface MultiTTSParams {
  script: string;
  speakers: SpeakerConfig[];
  crossfade_ms?: number;
}

export interface ElectronAPI {
  getServerPort: () => Promise<number>;
  generateTTS: (params: TTSParams) => Promise<void>;
  generateMultiTTS: (params: MultiTTSParams) => Promise<void>;
  onTTSChunk: (callback: (chunk: ArrayBuffer) => void) => void;
  onTTSComplete: (callback: () => void) => void;
  onTTSError: (callback: (error: string) => void) => void;
  removeAllListeners: () => void;
  // Voice management
  saveVoice: (params: { name: string; description: string; audioData: ArrayBuffer }) => Promise<SavedVoice>;
  getSavedVoices: () => Promise<SavedVoice[]>;
  deleteVoice: (id: string) => Promise<void>;
  // Dev tools
  toggleDevTools: () => Promise<void>;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}
