import { contextBridge, ipcRenderer } from 'electron';

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

contextBridge.exposeInMainWorld('electronAPI', {
  getServerPort: () => ipcRenderer.invoke('get-server-port'),
  generateTTS: (params: TTSParams) => ipcRenderer.invoke('tts:generate', params),
  generateMultiTTS: (params: MultiTTSParams) => ipcRenderer.invoke('tts:generate-multi', params),
  onTTSChunk: (callback: (chunk: ArrayBuffer) => void) => {
    ipcRenderer.on('tts:chunk', (_event, chunk) => callback(chunk));
  },
  onTTSComplete: (callback: () => void) => {
    ipcRenderer.on('tts:complete', () => callback());
  },
  onTTSError: (callback: (error: string) => void) => {
    ipcRenderer.on('tts:error', (_event, error) => callback(error));
  },
  removeAllListeners: () => {
    ipcRenderer.removeAllListeners('tts:chunk');
    ipcRenderer.removeAllListeners('tts:complete');
    ipcRenderer.removeAllListeners('tts:error');
  },
  // Voice management
  saveVoice: (params: { name: string; description: string; audioData: ArrayBuffer }) =>
    ipcRenderer.invoke('voice:save', params),
  getSavedVoices: () => ipcRenderer.invoke('voice:list'),
  deleteVoice: (id: string) => ipcRenderer.invoke('voice:delete', id),
  // Dev tools
  toggleDevTools: () => ipcRenderer.invoke('toggle-devtools'),
} as ElectronAPI);
