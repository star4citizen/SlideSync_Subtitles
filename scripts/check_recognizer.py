"""Check if a Speech-to-Text v2 recognizer exists and is accessible."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech


def check_recognizer(recognizer_name: str) -> None:
    """Check if recognizer exists and print its details."""
    # Extract location from recognizer name
    parts = recognizer_name.split("/")
    if "locations" not in parts:
        print(f"[ERROR] Invalid recognizer format: {recognizer_name}")
        print("[ERROR] Expected format: projects/{PROJECT_ID}/locations/{LOCATION}/recognizers/{RECOGNIZER_ID}")
        return
    
    location_idx = parts.index("locations")
    if location_idx + 1 >= len(parts):
        print("[ERROR] Location not found in recognizer name")
        return
    
    location = parts[location_idx + 1]
    
    # Create client
    if location == "global":
        client = SpeechClient(transport="grpc")
    else:
        client = SpeechClient(
            transport="grpc",
            client_options={"api_endpoint": f"{location}-speech.googleapis.com"},
        )
    
    print(f"[INFO] Checking recognizer: {recognizer_name}")
    print(f"[INFO] Location: {location}")
    print(f"[INFO] Endpoint: {location}-speech.googleapis.com" if location != "global" else "[INFO] Endpoint: global (default)")
    
    try:
        # Try to get the recognizer
        recognizer = client.get_recognizer(name=recognizer_name)
        print("[SUCCESS] Recognizer exists!")
        print(f"[INFO] Recognizer details:")
        print(f"  - Name: {recognizer.name}")
        print(f"  - Display name: {recognizer.display_name}")
        print(f"  - Model: {recognizer.model}")
        print(f"  - Language codes: {recognizer.language_codes}")
        print(f"  - State: {recognizer.state}")
        return True
    except Exception as e:
        error_str = str(e)
        print(f"[ERROR] Failed to get recognizer: {e}")
        
        if "404" in error_str or "not found" in error_str.lower():
            print("[ERROR] Recognizer does not exist!")
            print("[INFO] You need to create the recognizer first.")
            print("[INFO] Run: python scripts/create_recognizer.py")
        elif "403" in error_str or "permission" in error_str.lower():
            print("[ERROR] Permission denied!")
            print("[INFO] Check your service account permissions:")
            print("[INFO]   - Speech-to-Text API User role")
            print("[INFO]   - GOOGLE_APPLICATION_CREDENTIALS is set correctly")
        elif "500" in error_str or "internal" in error_str.lower():
            print("[ERROR] Google Cloud API internal error!")
            print("[INFO] This might be a temporary issue. Try again later.")
        else:
            print("[ERROR] Unknown error. Check your configuration.")
        
        return False


def main() -> int:
    """Main entry point."""
    # Load .env
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=repo_root / ".env")
    
    # Get recognizer from env or command line
    recognizer = os.getenv("GCP_RECOGNIZER", "")
    if len(sys.argv) > 1:
        recognizer = sys.argv[1]
    
    if not recognizer:
        print("[ERROR] No recognizer specified!")
        print("[INFO] Usage: python scripts/check_recognizer.py [RECOGNIZER_NAME]")
        print("[INFO] Or set GCP_RECOGNIZER in .env file")
        print("[INFO] Format: projects/{PROJECT_ID}/locations/{LOCATION}/recognizers/{RECOGNIZER_ID}")
        return 1
    
    success = check_recognizer(recognizer)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
