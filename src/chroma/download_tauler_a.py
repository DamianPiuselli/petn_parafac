import os
import urllib.request
import urllib.parse
import zipfile
import sys

def get_env_proxy():
    """
    Checks environment variables for proxy configurations.
    """
    for var in ['HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy']:
        val = os.environ.get(var)
        if val:
            return val
    return None

def get_proxy_handler():
    """
    Creates urllib.request.ProxyHandler if environment proxy variables are defined.
    """
    proxy_url = get_env_proxy()
    if proxy_url:
        print(f"Proxy detected in environment: {proxy_url}")
        return urllib.request.ProxyHandler({
            'http': proxy_url,
            'https': proxy_url
        })
    return None

def download_tauler_a():
    url = "http://www.cid.csic.es/homes/rtaqam/tmp/WEB_MCR/download/datasets/adataset.zip"
    dest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/chroma/tauler_a"))
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "adataset.zip")

    print(f"Downloading Real HPLC-DAD data set (A) from CSIC...")
    try:
        # Build urllib opener with proxy support
        proxy_handler = get_proxy_handler()
        if proxy_handler:
            opener = urllib.request.build_opener(proxy_handler)
        else:
            opener = urllib.request.build_opener()
            
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        
        with opener.open(req) as response, open(zip_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Successfully downloaded to: {zip_path}")
    except Exception as e:
        print(f"Error downloading: {e}")
        sys.exit(1)

    print("Extracting zip archive...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print(f"Extraction completed. Files saved to: {dest_dir}")
        os.remove(zip_path)
        print("Removed temporary zip archive.")
    except Exception as e:
        print(f"Error extracting zip archive: {e}")
        sys.exit(1)

    print("Download and extraction completed successfully.")

if __name__ == "__main__":
    download_tauler_a()
