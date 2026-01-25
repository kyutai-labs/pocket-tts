#!/usr/bin/env python3
"""
Generate API documentation for Pocket TTS using Sphinx.

This script builds the API documentation and creates a comprehensive
documentation site with auto-generated API reference.
"""

import os
import sys
import subprocess
from pathlib import Path


def install_dependencies():
    """Install documentation dependencies."""
    print("Installing documentation dependencies...")

    deps = [
        "sphinx",
        "sphinx-rtd-theme",
        "sphinx-autodoc-typehints",
        "sphinx.ext.napoleon",
        "myst-parser",
    ]

    for dep in deps:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", dep],
                check=True,
                capture_output=True,
            )
            print(f"✅ Installed {dep}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")


def build_docs():
    """Build the API documentation."""
    print("Building API documentation...")

    docs_dir = Path(__file__).parent
    api_dir = docs_dir / "api"
    build_dir = api_dir / "_build"

    # Create directories
    api_dir.mkdir(exist_ok=True)
    build_dir.mkdir(exist_ok=True)

    # Change to API directory for building
    original_cwd = os.getcwd()
    try:
        os.chdir(api_dir)

        # Clean previous build
        if build_dir.exists():
            import shutil

            shutil.rmtree(build_dir)

        # Build HTML documentation
        cmd = [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            "-c",
            str(api_dir),  # Use api_dir as config directory
            ".",
            "_build/html",
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Documentation built successfully!")
            print(f"📚 Documentation available at: {build_dir / 'html' / 'index.html'}")
            return True
        else:
            print("❌ Documentation build failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

    finally:
        os.chdir(original_cwd)


def verify_docs():
    """Verify that documentation was built correctly."""
    print("Verifying documentation...")

    docs_dir = Path(__file__).parent
    index_file = docs_dir / "api" / "_build" / "html" / "index.html"

    if index_file.exists():
        print("✅ Index file exists")

        # Check file size (should be substantial)
        size = index_file.stat().st_size
        print(f"📄 Index file size: {size:,} bytes")

        if size > 1000:  # At least 1KB
            print("✅ Documentation appears to be properly generated")
            return True
        else:
            print("❌ Documentation file too small")
            return False
    else:
        print("❌ Index file not found")
        return False


def main():
    """Main function to generate documentation."""
    print("🚀 Pocket TTS Documentation Generator")
    print("=" * 50)

    # Install dependencies
    install_dependencies()

    # Build documentation
    if build_docs():
        # Verify documentation
        if verify_docs():
            print("\n🎉 Documentation generation completed successfully!")
            print("\nTo view the documentation:")
            print("1. Open the following file in your browser:")
            print(
                f"   file://{Path(__file__).parent / 'api' / '_build' / 'html' / 'index.html'}"
            )
            print("\n2. Or serve it locally:")
            print("   cd docs/api/_build/html")
            print("   python -m http.server 8000")
            print("   # Then visit http://localhost:8000")
        else:
            print("\n❌ Documentation verification failed")
            sys.exit(1)
    else:
        print("\n❌ Documentation build failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
