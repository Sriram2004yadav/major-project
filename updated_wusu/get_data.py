import os
import tarfile
from lightning_sdk import Studio

# The name of the OTHER studio where the data lives
SOURCE_STUDIO = "continuous-wusu"

print(f"1. Connecting to {SOURCE_STUDIO}...")
s = Studio(name=SOURCE_STUDIO)

print("2. Compressing folder on REMOTE machine (using tar)...")
# We use 'tar' because it is built-in on all Linux systems
s.run("tar -czf WUSU_transfer.tar.gz WUSU_processed")

print("3. Downloading the compressed file...")
s.download_file("WUSU_transfer.tar.gz", "WUSU_transfer.tar.gz")

print("4. Extracting locally...")
with tarfile.open("WUSU_transfer.tar.gz", "r:gz") as tar:
    tar.extractall()

# Cleanup
os.remove("WUSU_transfer.tar.gz")
# Optional: Clean up the remote file too so it doesn't waste space there
s.run("rm WUSU_transfer.tar.gz")

print("Success! 'WUSU_processed' is ready.")

print("Success! 'WUSU_processed' is now in your folder.")