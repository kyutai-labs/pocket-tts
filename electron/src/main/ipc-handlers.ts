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

export function registerIpcHandlers(pythonServer: PythonServer, voiceManager: { getVoiceFilePath: (id: string) => string | null }) {
  ipcMain.handle('tts:generate', async (event: IpcMainInvokeEvent, params: TTSParams) => {
    const { text, voiceUrl, voiceFile, savedVoiceId } = params;
    const sender = event.sender;

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
}
