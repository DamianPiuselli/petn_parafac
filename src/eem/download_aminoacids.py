"""
Utility script to download and extract the experimental Amino Acids EEM dataset.
Saves the raw file to data/raw/amino.mat.
"""
import urllib.request
import zipfile
import io
import os
from scipy.io import loadmat

def download_and_extract():
    url = 'https://sid.erda.dk/share_redirect/AWqL5T6JCZ'
    dest_dir = 'data/raw'
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Downloading from {url}...")
    # Fetch nested zip data with User-Agent header
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        outer_zip_data = response.read()
    
    print("Extracting outer zip...")
    outer_zip = zipfile.ZipFile(io.BytesIO(outer_zip_data))
    inner_zip_bytes = outer_zip.read('Amino_Acid_fluo.zip')
    
    print("Extracting inner zip...")
    inner_zip = zipfile.ZipFile(io.BytesIO(inner_zip_bytes))
    amino_mat_bytes = inner_zip.read('amino.mat')
    
    dest_path = os.path.join(dest_dir, 'amino.mat')
    with open(dest_path, 'wb') as f:
        f.write(amino_mat_bytes)
    
    print(f"Successfully saved to {dest_path}")
    
    # Verify that the mat file can be read and check its shape parameters
    mat = loadmat(dest_path)
    print("Keys in amino.mat:", [k for k in mat.keys() if not k.startswith('__')])
    print("DimX (dimensions):", mat['DimX'])
    print("ExAx (excitation wavelengths):", mat['ExAx'].squeeze())
    print("EmAx (emission wavelengths):", mat['EmAx'].squeeze())

if __name__ == '__main__':
    download_and_extract()
