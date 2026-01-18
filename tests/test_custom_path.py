import os
import unittest
from pathlib import Path

from pocket_tts.utils.utils import download_if_necessary


class TestCustomPath(unittest.TestCase):
    def test_local_path_override(self):
        # Setup fake local file
        local_dir = Path("./temp_models")
        local_dir.mkdir(exist_ok=True)
        # Create the specific directory structure that matches the HF repo ID if needed,
        # or simplified if logic allows.
        # Based on current implementation, download_if_necessary handles "hf://" by splitting.
        # hf://kyutai/pocket-tts/embeddings/cosette.safetensors
        # repo_id: kyutai/pocket-tts
        # filename: embeddings/cosette.safetensors

        # If we implement simple recursive search or mimicking structure, let's test that.
        # For now, let's assume we want to map flat structure or mirror HF.
        # Let's try to mimic the filename part first.

        target_file = local_dir / "embeddings" / "cosette.safetensors"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.touch()

        # Set env var
        os.environ["POCKET_TTS_LOCAL_MODELS_PATH"] = str(local_dir.absolute())

        try:
            # Test
            url = "hf://kyutai/pocket-tts/embeddings/cosette.safetensors"
            result = download_if_necessary(url)

            # Verify
            # The result should be the absolute path to our local file
            self.assertEqual(result.resolve(), target_file.resolve())
            print("Successfully resolved to local file!")

        finally:
            # Cleanup
            if target_file.exists():
                target_file.unlink()
            if target_file.parent.exists():
                target_file.parent.rmdir()
            if local_dir.exists():
                # might fail if other files are there, but for this test it's fine
                import shutil

                shutil.rmtree(local_dir)
            if "POCKET_TTS_LOCAL_MODELS_PATH" in os.environ:
                del os.environ["POCKET_TTS_LOCAL_MODELS_PATH"]


if __name__ == "__main__":
    unittest.main()
