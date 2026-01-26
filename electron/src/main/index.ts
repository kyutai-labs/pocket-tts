import { app, BrowserWindow, ipcMain } from 'electron';
import * as path from 'path';
import { PythonServer } from './python-server';
import { registerIpcHandlers } from './ipc-handlers';
import { registerVoiceHandlers, getVoiceManager } from './voice-manager';

let mainWindow: BrowserWindow | null = null;
let pythonServer: PythonServer | null = null;

function isDev(): boolean {
  return !app.isPackaged;
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
  } else {
    const htmlPath = path.join(__dirname, '../renderer/index.html');
    console.log('Loading HTML from:', htmlPath);
    console.log('__dirname:', __dirname);
    mainWindow.loadFile(htmlPath);
  }

  // DevTools off by default - can be toggled via IPC

  // Log any load failures
  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
    console.error('Failed to load:', errorCode, errorDescription);
  });

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
  registerVoiceHandlers();

  // Create window first so user sees the app
  await createWindow();

  // Register IPC handlers with getter to access current server state
  registerIpcHandlers(() => pythonServer, getVoiceManager());

  try {
    await startPythonServer();
    console.log('Python server started successfully');
  } catch (error) {
    console.error('Failed to start Python server:', error);
    // Show error in the window after it's ready
    mainWindow?.webContents.once('did-finish-load', () => {
      mainWindow?.webContents.executeJavaScript(
        `console.error('TTS server failed to start'); alert('Failed to start TTS server: ${String(error).replace(/'/g, "\\'")}\\n\\nMake sure the Python server is bundled or run in dev mode.')`
      );
    });
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

ipcMain.handle('toggle-devtools', () => {
  if (mainWindow) {
    if (mainWindow.webContents.isDevToolsOpened()) {
      mainWindow.webContents.closeDevTools();
    } else {
      mainWindow.webContents.openDevTools();
    }
  }
});
