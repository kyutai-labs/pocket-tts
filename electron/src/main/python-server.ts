import { spawn, ChildProcess } from 'child_process';
import { app } from 'electron';
import * as path from 'path';
import * as net from 'net';

export class PythonServer {
  private process: ChildProcess | null = null;
  public port: number = 0;

  private get isDev(): boolean {
    return !app.isPackaged;
  }

  async start(): Promise<void> {
    this.port = await this.findAvailablePort();

    const command = this.getCommand();
    const args = this.getArgs();

    const cwd = this.getWorkingDirectory();
    console.log(`Starting Python server: ${command} ${args.join(' ')} in ${cwd}`);

    this.process = spawn(command, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env },
      cwd,
    });

    this.process.stdout?.on('data', (data) => {
      console.log(`[Python]: ${data.toString().trim()}`);
    });

    this.process.stderr?.on('data', (data) => {
      console.error(`[Python Error]: ${data.toString().trim()}`);
    });

    this.process.on('error', (error) => {
      console.error('Failed to start Python server:', error);
    });

    this.process.on('exit', (code) => {
      console.log(`Python server exited with code ${code}`);
    });

    await this.waitForReady();
  }

  async stop(): Promise<void> {
    if (this.process) {
      this.process.kill('SIGTERM');
      this.process = null;
    }
  }

  private getCommand(): string {
    if (this.isDev) {
      return 'uv';
    }
    // Production: use bundled Python
    const resourcePath = process.resourcesPath;
    if (process.platform === 'win32') {
      return path.join(resourcePath, 'pocket-tts-server', 'pocket-tts-server.exe');
    }
    return path.join(resourcePath, 'pocket-tts-server', 'pocket-tts-server');
  }

  private getArgs(): string[] {
    if (this.isDev) {
      return ['run', 'pocket-tts', 'serve', '--port', this.port.toString()];
    }
    // Note: entry script already adds "serve" command, so just pass options
    return ['--port', this.port.toString()];
  }

  private getWorkingDirectory(): string {
    if (this.isDev) {
      // In dev mode, run from the pocket-tts root directory (parent of electron/)
      return path.resolve(__dirname, '..', '..', '..');
    }
    // In production, run from the app resources directory
    return process.resourcesPath;
  }

  private async findAvailablePort(): Promise<number> {
    return new Promise((resolve, reject) => {
      const server = net.createServer();
      server.listen(0, () => {
        const address = server.address();
        if (address && typeof address === 'object') {
          const port = address.port;
          server.close(() => resolve(port));
        } else {
          reject(new Error('Failed to get port'));
        }
      });
      server.on('error', reject);
    });
  }

  private async waitForReady(timeout = 30000): Promise<void> {
    const startTime = Date.now();
    const healthUrl = `http://localhost:${this.port}/health`;

    while (Date.now() - startTime < timeout) {
      try {
        const response = await fetch(healthUrl);
        if (response.ok) {
          console.log('Python server is ready');
          return;
        }
      } catch {
        // Server not ready yet
      }
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    throw new Error('Python server failed to start within timeout');
  }
}
