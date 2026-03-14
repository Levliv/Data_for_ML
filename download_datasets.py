import os
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
import shutil
import json

BASE_SAVE_PATH = "/Users/lev/Documents/ml_for_data/data/raw"

def process_huggingface(dataset_info):
    dataset_id = dataset_info["id"]
    slug = dataset_id.replace("/", "_")
    save_dir = os.path.join(BASE_SAVE_PATH, "huggingface", slug)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Downloading HF dataset: {dataset_id}")
    ds_dict = load_dataset(dataset_id)
    
    dfs = []
    for split in ds_dict.keys():
        df = ds_dict[split].to_pandas()
        df["_source"] = "huggingface"
        df["_dataset_id"] = dataset_id
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_parquet(os.path.join(save_dir, "data.parquet"))

def download_datasets(datasets):
    success = []
    failed = []
    
    for ds in datasets:
        try:
            if ds["source"] == "huggingface":
                process_huggingface(ds)
            else:
                print(f"Unsupported source: {ds['source']}")
                raise ValueError("Source not implemented")
            
            success.append(ds["id"])
        except Exception as e:
            print(f"Failed to process {ds['id']}: {e}")
            failed.append(ds["id"])
            
    print("\n--- Final Summary ---")
    print(f"Successful: {', '.join(success) if success else 'None'}")
    print(f"Failed: {', '.join(failed) if failed else 'None'}")

if __name__ == "__main__":
    datasets_to_process = [
      {
        "id": "maniteja7463/python_programming_questions",
        "name": "Python Programming Questions",
        "source": "huggingface",
        "url": "https://huggingface.co/datasets/maniteja7463/python_programming_questions",
        "downloads": 18,
        "likes": 0,
        "size_category": "10K<n<100K",
        "license": "unknown",
        "description": "A collection of Python-specific programming questions. Suitable for fine-tuning LLMs on coding interview preparation or language-specific tasks.",
        "ml_tasks": [
          "text-generation",
          "question-answering"
        ],
        "relevance_score": 8
      }
    ]
    
    download_datasets(datasets_to_process)