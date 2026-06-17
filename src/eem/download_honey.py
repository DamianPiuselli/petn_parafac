"""
Utility script to download and extract the experimental Honey EEM dataset.
Saves the raw files to data/raw/honey/.
"""
import urllib.request
import zipfile
import io
import os
from scipy.io import loadmat

def download_and_extract():
    url = 'https://sid.erda.dk/share_redirect/fGlqQtWWut/HoneyEEM.zip'
    dest_dir = 'data/raw/honey'
    os.makedirs(dest_dir, exist_ok=True)
    
    dest_path = os.path.join(dest_dir, 'HoneyEEM.mat')
    
    if not os.path.exists(dest_path):
        print(f"Downloading from {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            zip_data = response.read()
        
        print("Extracting zip...")
        zip_file = zipfile.ZipFile(io.BytesIO(zip_data))
        zip_file.extractall(dest_dir)
        print(f"Successfully extracted all files to {dest_dir}")
    else:
        print(f"File already exists at {dest_path}")
        
    # Verify that the mat file can be read and check its shape parameters
    mat = loadmat(dest_path)
    print("Keys in HoneyEEM.mat:", [k for k in mat.keys() if not k.startswith('__')])
    
    for key in mat.keys():
        if not key.startswith('__'):
            val = mat[key]
            # If it's a numpy array, print its shape, else its type
            if hasattr(val, 'shape'):
                print(f"Shape of {key}: {val.shape}")
            else:
                print(f"Type of {key}: {type(val)}")

if __name__ == '__main__':
    download_and_extract()
