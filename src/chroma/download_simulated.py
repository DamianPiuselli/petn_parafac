import os
import urllib.request
import zipfile
import sys

def download_simulated():
    url = "https://sid.erda.dk/share_redirect/eQxBjzieMJ/Simulated%20GCMS%20data.zip"
    dest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/chroma/simulated"))
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "Simulated_GCMS_data.zip")

    print(f"Downloading UCPH Simulated GC-MS Benchmarks...")
    print(f"Source URL: {url}")
    print(f"Destination: {zip_path}")

    # Helper function to show progress
    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            s = f"\rDownloading: {percent:3.1f}% [{readsofar}/{totalsize} bytes]"
            sys.stdout.write(s)
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\rRead {readsofar} bytes")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, zip_path, reporthook)
        print("\nDownload completed successfully.")
    except Exception as e:
        print(f"\nError downloading the file: {e}")
        sys.exit(1)

    print("Unpacking zip archive...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print(f"Extraction completed. Files saved to: {dest_dir}")
        # Remove the zip file to save space if needed
        os.remove(zip_path)
        print("Removed temporary zip archive.")
    except Exception as e:
        print(f"Error extracting zip archive: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_simulated()
