import { app, BrowserWindow, ipcMain } from 'electron';
import * as path from 'path';
import { PythonServer } from './python-server';
import { registerIpcHandlers } from './ipc-handlers';
import { registerVoiceHandlers, getVoiceManager } from './voice-manager';

let mainWindow: BrowserWindow | null = null;
let pythonServer: PythonServer | null = null;

function isDev(): boolean {
  return process.env.NODE_ENV !== 'production' || !app.isPackaged;
}

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 900,
    height: 750,
    minWidth: 600,
    minHeight: 500,
    backgroundColor: '#1a1a2e',
    titleBarStyle: 'hiddenInset',
    webPreferences: {
      preload: path.join(__dirname, '../preload/preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (isDev()) {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

async function startPythonServer() {
  pythonServer = new PythonServer();
  try {
    await pythonServer.start();
    console.log(`Python server started on port ${pythonServer.port}`);
  } catch (error) {
    console.error('Failed to start Python server:', error);
    throw error;
  }
}

app.whenReady().then(async () => {
  try {
    registerVoiceHandlers();
    await startPythonServer();
    registerIpcHandlers(pythonServer!, getVoiceManager());
    await createWindow();
  } catch (error) {
    console.error('Failed to initialize app:', error);
    app.quit();
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', async () => {
  if (pythonServer) {
    await pythonServer.stop();
  }
});

ipcMain.handle('get-server-port', () => {
  return pythonServer?.port ?? 8000;
});
