"""Create a Speech-to-Text v2 recognizer resource."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech


def create_recognizer(
    project_id: str,
    location: str,
    recognizer_id: str,
    language_code: str = "ko-KR",
    model: str = "long",
    display_name: str = "",
) -> str:
    """Create a Speech-to-Text v2 recognizer."""
    # Create client
    if location == "global":
        client = SpeechClient(transport="grpc")
        parent = f"projects/{project_id}/locations/{location}"
    else:
        client = SpeechClient(
            transport="grpc",
            client_options={"api_endpoint": f"{location}-speech.googleapis.com"},
        )
        parent = f"projects/{project_id}/locations/{location}"
    
    if not display_name:
        display_name = recognizer_id
    
    print(f"[INFO] Creating recognizer in: {parent}")
    print(f"[INFO] Recognizer ID: {recognizer_id}")
    print(f"[INFO] Display name: {display_name}")
    print(f"[INFO] Language: {language_code}")
    print(f"[INFO] Model: {model}")
    
    # Create recognizer request
    recognizer = cloud_speech.Recognizer(
        display_name=display_name,
        model=model,
        language_codes=[language_code],
    )
    
    request = cloud_speech.CreateRecognizerRequest(
        parent=parent,
        recognizer=recognizer,
        recognizer_id=recognizer_id,
    )
    
    try:
        operation = client.create_recognizer(request=request)
        print("[INFO] Recognizer creation started. Waiting for completion...")
        
        # Wait for operation to complete
        response = operation.result(timeout=60)
        recognizer_name = response.name
        print(f"[SUCCESS] Recognizer created successfully!")
        print(f"[INFO] Full name: {recognizer_name}")
        return recognizer_name
    except Exception as e:
        error_str = str(e)
        print(f"[ERROR] Failed to create recognizer: {e}")
        
        if "409" in error_str or "already exists" in error_str.lower():
            print("[WARN] Recognizer already exists!")
            recognizer_name = f"{parent}/recognizers/{recognizer_id}"
            print(f"[INFO] Using existing recognizer: {recognizer_name}")
            return recognizer_name
        elif "403" in error_str or "permission" in error_str.lower():
            print("[ERROR] Permission denied!")
            print("[INFO] Check your service account permissions:")
            print("[INFO]   - Speech-to-Text Admin role or")
            print("[INFO]   - Speech-to-Text Editor role")
            print("[INFO]   - GOOGLE_APPLICATION_CREDENTIALS is set correctly")
        elif "400" in error_str:
            print("[ERROR] Invalid request!")
            print("[INFO] Check your parameters (location, project_id, etc.)")
        else:
            print("[ERROR] Unknown error. Check your configuration.")
        
        raise


def main() -> int:
    """Main entry point."""
    # Load .env
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=repo_root / ".env")
    
    # Get parameters from env or command line
    project_id = os.getenv("GCP_PROJECT_ID", "")
    location = os.getenv("GCP_LOCATION", "asia-northeast1")
    recognizer_id = os.getenv("GCP_RECOGNIZER_ID", "krspeechtotext")
    language_code = os.getenv("GCP_LANGUAGE_CODE", "ko-KR")
    model = os.getenv("GCP_STT_MODEL", "long")
    
    # Override with command line args if provided
    if len(sys.argv) > 1:
        project_id = sys.argv[1]
    if len(sys.argv) > 2:
        location = sys.argv[2]
    if len(sys.argv) > 3:
        recognizer_id = sys.argv[3]
    
    if not project_id:
        print("[ERROR] No project ID specified!")
        print("[INFO] Usage: python scripts/create_recognizer.py [PROJECT_ID] [LOCATION] [RECOGNIZER_ID]")
        print("[INFO] Or set GCP_PROJECT_ID in .env file")
        return 1
    
    try:
        recognizer_name = create_recognizer(
            project_id=project_id,
            location=location,
            recognizer_id=recognizer_id,
            language_code=language_code,
            model=model,
        )
        print("\n[INFO] Add this to your .env file:")
        print(f"GCP_RECOGNIZER={recognizer_name}")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Failed to create recognizer: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
