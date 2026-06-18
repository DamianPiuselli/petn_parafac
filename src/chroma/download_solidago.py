import os
import urllib.request
import sys

def download_solidago():
    base_url = "https://raw.githubusercontent.com/ethanbass/chromatographR/master"
    files = {
        "Sa.RData": f"{base_url}/data/Sa.RData",
        "Sa_pr.RData": f"{base_url}/data/Sa_pr.RData",
        "Sa_warp.RData": f"{base_url}/data/Sa_warp.RData",
        "pk_tab.RData": f"{base_url}/data/pk_tab.RData",
        "Sa_metadata.csv": f"{base_url}/inst/extdata/Sa_metadata.csv"
    }

    dest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/chroma/solidago"))
    os.makedirs(dest_dir, exist_ok=True)

    print("Downloading Solidago altissima Root Extracts (HPLC-DAD) files from GitHub...")
    for filename, url in files.items():
        dest_path = os.path.join(dest_dir, filename)
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, dest_path)
            print(f"Successfully downloaded to: {dest_path}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            sys.exit(1)

    print("All Solidago files downloaded successfully.")

if __name__ == "__main__":
    download_solidago()
