import os
import tarfile
from lightning_sdk import Studio

SOURCE_STUDIO = "final_sn7" 
REMOTE_PATH = "spacenet7"

print(f"🚀 Connecting to {SOURCE_STUDIO}...")
s = Studio(name=SOURCE_STUDIO)

# Use -cf (no compression) for max speed. 
# We use .tar instead of .tar.gz
print("📦 Bundling SpaceNet 7 data (No compression for speed)...")
try:
    # Removing the 'z' flag makes this 5-10x faster on the CPU
    s.run(f"tar -cf sn7_fast_transfer.tar -C {REMOTE_PATH} .")
    
    print("📡 Downloading uncompressed bundle...")
    # It's a larger file, but total time (Bundle + Download) is usually much lower
    s.download_file("sn7_fast_transfer.tar", "sn7_fast_transfer.tar")

    print("🔓 Extracting locally...")
    os.makedirs("spacenet7", exist_ok=True)
    with tarfile.open("sn7_fast_transfer.tar", "r") as tar:
        tar.extractall(path="spacenet7")

    # Cleanup
    print("🧹 Cleaning up temporary files...")
    os.remove("sn7_fast_transfer.tar")
    s.run("rm sn7_fast_transfer.tar")
    print("✅ Done! Data is ready in 'spacenet7' folder.")
    
except Exception as e:
    print(f"❌ Error: {e}")