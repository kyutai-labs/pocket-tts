import { ipcMain, IpcMainInvokeEvent } from 'electron';
import * as fs from 'fs';
import { PythonServer } from './python-server';
import type { VoiceManager } from './voice-manager';

interface TTSParams {
  text: string;
  voiceUrl?: string;
  voiceFile?: ArrayBuffer;
  savedVoiceId?: string;
}

interface SpeakerConfig {
  name: string;
  voice_source: string;
  voice_data: string | null;
  seed: number | null;
}

interface MultiTTSParams {
  script: string;
  speakers: SpeakerConfig[];
  crossfade_ms?: number;
}

export function registerIpcHandlers(
  getPythonServer: () => PythonServer | null,
  voiceManager: { getVoiceFilePath: (id: string) => string | null }
) {
  ipcMain.handle('tts:generate', async (event: IpcMainInvokeEvent, params: TTSParams) => {
    const { text, voiceUrl, voiceFile, savedVoiceId } = params;
    const sender = event.sender;

    const pythonServer = getPythonServer();
    if (!pythonServer || !pythonServer.port) {
      sender.send('tts:error', 'TTS server is not running. Please restart the app or check the Python server.');
      return;
    }

    try {
      const formData = new FormData();
      formData.append('text', text);

      if (voiceFile) {
        const blob = new Blob([voiceFile], { type: 'audio/wav' });
        formData.append('voice_wav', blob, 'voice.wav');
      } else if (savedVoiceId) {
        // Load saved voice from file system
        const filePath = voiceManager.getVoiceFilePath(savedVoiceId);
        if (filePath && fs.existsSync(filePath)) {
          const buffer = fs.readFileSync(filePath);
          const blob = new Blob([buffer], { type: 'audio/wav' });
          formData.append('voice_wav', blob, 'voice.wav');
        } else {
          throw new Error('Saved voice file not found');
        }
      } else if (voiceUrl) {
        formData.append('voice_url', voiceUrl);
      }

      const response = await fetch(`http://localhost:${pythonServer.port}/tts`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server error: ${response.status} - ${errorText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        if (value && !sender.isDestroyed()) {
          sender.send('tts:chunk', value.buffer);
        }
      }

      if (!sender.isDestroyed()) {
        sender.send('tts:complete');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      if (!sender.isDestroyed()) {
        sender.send('tts:error', errorMessage);
      }
    }
  });

  // Multi-Talk TTS handler
  ipcMain.handle('tts:generate-multi', async (event: IpcMainInvokeEvent, params: MultiTTSParams) => {
    const { script, speakers, crossfade_ms = 100 } = params;
    const sender = event.sender;

    const pythonServer = getPythonServer();
    if (!pythonServer || !pythonServer.port) {
      sender.send('tts:error', 'TTS server is not running. Please restart the app or check the Python server.');
      return;
    }

    try {
      // Resolve saved voices to base64 data
      const resolvedSpeakers = speakers.map((speaker) => {
        if (speaker.voice_source.startsWith('saved:')) {
          const savedVoiceId = speaker.voice_source.replace('saved:', '');
          const filePath = voiceManager.getVoiceFilePath(savedVoiceId);
          if (filePath && fs.existsSync(filePath)) {
            const buffer = fs.readFileSync(filePath);
            const base64Data = buffer.toString('base64');
            return {
              ...speaker,
              voice_source: 'uploaded',
              voice_data: base64Data,
            };
          } else {
            throw new Error(`Saved voice file not found for speaker "${speaker.name}"`);
          }
        }
        return speaker;
      });

      const formData = new FormData();
      formData.append('script', script);
      formData.append('speakers', JSON.stringify(resolvedSpeakers));
      formData.append('crossfade_ms', crossfade_ms.toString());

      const response = await fetch(`http://localhost:${pythonServer.port}/multi-tts`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        try {
          const errorJson = JSON.parse(errorText);
          throw new Error(errorJson.detail || `Server error: ${response.status}`);
        } catch {
          throw new Error(`Server error: ${response.status} - ${errorText}`);
        }
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        if (value && !sender.isDestroyed()) {
          sender.send('tts:chunk', value.buffer);
        }
      }

      if (!sender.isDestroyed()) {
        sender.send('tts:complete');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      if (!sender.isDestroyed()) {
        sender.send('tts:error', errorMessage);
      }
    }
  });
}
