import os
import requests
import pandas as pd
import pyarrow as pa
from datasets import load_dataset
from pathlib import Path
import tempfile
import shutil
import zipfile

DATASETS = [
    {
        "id": "maniteja7463/python_programming_questions",
        "name": "Python Programming Questions",
        "source": "huggingface",
        "url": "https://huggingface.co/datasets/maniteja7463/python_programming_questions",
        "downloads": 18,
        "likes": 0,
        "size_category": "10K<n<100K",
        "license": "unknown",
        "description": "A collection of Python-specific programming questions. Suitable for training models on code generation or question-answering tasks.",
        "ml_tasks": ["text-generation", "question-answering"],
        "relevance_score": 9
    }
]

BASE_DIR = Path("/Users/lev/Documents/ml_for_data/data/raw")

def process_huggingface(ds_meta):
    ds_id = ds_meta["id"]
    slug = ds_id.split("/")[-1]
    save_path = BASE_DIR / "huggingface" / slug / "data.parquet"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset(ds_id)
        dfs = []
        for split in ds.keys():
            df = ds[split].to_pandas()
            df["_source"] = "huggingface"
            df["_dataset_id"] = ds_id
            dfs.append(df)
        
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_parquet(save_path)
        return True, None
    except Exception as e:
        return False, str(e)

def process_kaggle(ds_meta):
    # Requires 'kaggle' cli to be authenticated
    ds_id = ds_meta["id"]
    slug = ds_id.split("/")[-1]
    save_path = BASE_DIR / "kaggle" / slug / "data.parquet"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            os.system(f"kaggle datasets download -d {ds_id} -p {tmp_dir} --unzip")
            dfs = []
            for root, _, files in os.walk(tmp_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    if f.endswith('.csv'): df = pd.read_csv(fp)
                    elif f.endswith('.json'): df = pd.read_json(fp)
                    elif f.endswith('.parquet'): df = pd.read_parquet(fp)
                    else: continue
                    
                    df["_source"] = "kaggle"
                    df["_dataset_id"] = ds_id
                    dfs.append(df)
            
            pd.concat(dfs, ignore_index=True).to_parquet(save_path)
            return True, None
        except Exception as e:
            return False, str(e)

def process_zenodo(ds_meta):
    record_id = ds_meta["url"].rstrip("/").split("/")[-1]
    slug = record_id
    save_path = BASE_DIR / "zenodo" / slug / "data.parquet"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        resp = requests.get(f"https://zenodo.org/api/records/{record_id}").json()
        dfs = []
        for file in resp["files"]:
            download_url = f"{file['links']['self']}/content"
            with tempfile.NamedTemporaryFile() as tmp:
                with requests.get(download_url, stream=True) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=8192):
                        tmp.write(chunk)
                tmp.flush()
                
                if file['key'].endswith('.csv'): df = pd.read_csv(tmp.name)
                elif file['key'].endswith('.json'): df = pd.read_json(tmp.name)
                elif file['key'].endswith('.parquet'): df = pd.read_parquet(tmp.name)
                else: continue
                
                df["_source"] = "zenodo"
                df["_dataset_id"] = ds_meta["id"]
                dfs.append(df)
        
        pd.concat(dfs, ignore_index=True).to_parquet(save_path)
        return True, None
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    results = {"success": [], "failed": []}
    
    for ds in DATASETS:
        print(f"Processing {ds['id']}...")
        source = ds['source']
        
        if source == "huggingface":
            ok, err = process_huggingface(ds)
        elif source == "kaggle":
            ok, err = process_kaggle(ds)
        elif source == "zenodo":
            ok, err = process_zenodo(ds)
        else:
            ok, err = False, "Unsupported source"
            
        if ok:
            results["success"].append(ds['id'])
            print(f"Successfully processed {ds['id']}")
        else:
            results["failed"].append((ds['id'], err))
            print(f"Failed {ds['id']}: {err}")

    print("\n--- Summary ---")
    print(f"Successful: {len(results['success'])}")
    for s in results['success']: print(f" - {s}")
    print(f"Failed: {len(results['failed'])}")
    for f in results['failed']: print(f" - {f[0]}: {f[1]}")