import os
import requests
import subprocess
import shutil
from pathlib import Path
from datasets import load_dataset

def download_zenodo(record_id, save_path):
    record_id = record_id.split(":")[-1]
    api_url = f"https://zenodo.org/api/records/{record_id}"
    
    response = requests.get(api_url)
    response.raise_for_status()
    
    files = response.json()["files"]
    os.makedirs(save_path, exist_ok=True)
    
    for file in files:
        filename = file["key"]
        url = file["links"]["content"]
        file_path = os.path.join(save_path, filename)
        
        print(f"Downloading {filename}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
    return True

def download_kaggle(dataset_id, save_path):
    os.makedirs(save_path, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset_id, "-p", save_path, "--unzip"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(result.stderr)
    return True

def main():
    base_dir = "/Users/lev/Documents/ml_for_data/data/raw"
    datasets_to_download = [
        {
            "id": "zenodo:4415593",
            "name": "SOTorrent Dataset",
            "source": "zenodo",
            "slug": "sotorrent"
        }
    ]
    
    summary = []
    
    for ds in datasets_to_download:
        source = ds["source"]
        slug = ds["slug"]
        save_path = os.path.join(base_dir, source, slug)
        
        print(f"--- Processing {ds['name']} ---")
        try:
            if source == "zenodo":
                download_zenodo(ds["id"], save_path)
            elif source == "kaggle":
                download_kaggle(ds["id"], save_path)
            else:
                print(f"Unsupported source: {source}")
                continue
            
            summary.append((ds['name'], "Success"))
            print(f"Successfully downloaded to {save_path}")
        except Exception as e:
            summary.append((ds['name'], f"Failed: {str(e)}"))
            print(f"Error downloading {ds['name']}: {e}")
            
    print("\n--- Final Summary ---")
    for name, status in summary:
        print(f"{name}: {status}")

if __name__ == "__main__":
    main()