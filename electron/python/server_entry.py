"""Entry point for PyInstaller bundled server."""
import sys


def main():
    # Import here to ensure all modules are collected by PyInstaller
    from pocket_tts.main import cli_app

    # Run the serve command with provided args
    sys.argv = ["pocket-tts", "serve"] + sys.argv[1:]
    cli_app()


if __name__ == "__main__":
    main()
