import os
import sys
import tarfile
import subprocess

# --- CONFIGURATION ---
# Updated with your specific Google Drive File ID
GDRIVE_FILE_ID = "18hq0FqlNGQ3dQ6HnlDtzH9lUBD5mFknM" 

# Target extraction path based on repository structure
TARGET_DIR = os.path.join("sign_to_speech", "01_data_collection_raspi")
TAR_FILENAME = "training_data.tar"
TAR_FILEPATH = os.path.join(TARGET_DIR, TAR_FILENAME)

def install_gdown():
    """Installs gdown if it is not already available."""
    try:
        import gdown
    except ImportError:
        print("Required package 'gdown' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])

def main():
    install_gdown()
    import gdown

    # Ensure the target directory exists
    os.makedirs(TARGET_DIR, exist_ok=True)

    # 1. Download the file
    print(f"\n[1/3] Downloading dataset from Google Drive...")
    gdown_url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    
    try:
        gdown.download(gdown_url, TAR_FILEPATH, quiet=False)
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

    # 2. Extract the tar file
    if os.path.exists(TAR_FILEPATH):
        print(f"\n[2/3] Extracting {TAR_FILENAME} into {TARGET_DIR}...")
        try:
            with tarfile.open(TAR_FILEPATH, "r") as tar:
                # Standard extraction
                tar.extractall(path=TARGET_DIR)
            print("Extraction successful.")
        except Exception as e:
            print(f"Extraction failed: {e}")
            sys.exit(1)

        # 3. Clean up the compressed file
        print(f"\n[3/3] Cleaning up temporary files...")
        os.remove(TAR_FILEPATH)
        print("\nDataset setup is complete!")
    else:
        print("\nError: Download failed, tar file not found.")

if __name__ == "__main__":
    main()