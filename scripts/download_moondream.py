#!/usr/bin/env python3
# Use: ./venv/bin/python3 scripts/download_moondream.py
"""
Pre-download Moondream2 model to avoid delay during first run.
This is optional - the model will auto-download on first use if not pre-downloaded.
"""

import sys
from huggingface_hub import snapshot_download

def main():
    model_id = "vikhyatk/moondream2"
    revision = "2025-06-21"
    
    print(f"Downloading Moondream2 model...")
    print(f"Model: {model_id}")
    print(f"Revision: {revision}")
    print(f"Size: ~3.7 GB")
    print(f"Location: ~/.cache/huggingface/")
    print("")
    
    try:
        path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            resume_download=True,  # Resume if interrupted
        )
        
        print("")
        print(f"✅ Download complete!")
        print(f"Model cached at: {path}")
        print("")
        print("You can now run 'cargo run --release' without waiting for model download.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Download interrupted. Run this script again to resume.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("Check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
