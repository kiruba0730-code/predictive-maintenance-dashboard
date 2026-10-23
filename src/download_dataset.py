"""
Download script for the AI4I 2020 Predictive Maintenance Dataset from UCI Machine Learning Repository.
Dataset URL: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
Direct static link: https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip
"""

import io
import os
import sys
import zipfile
import requests
import pandas as pd

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
GITHUB_RAW_MIRROR = "https://raw.githubusercontent.com/marshald/Predictive-Maintenance-AI4I-2020/main/ai4i2020.csv"
KAGGLE_MIRROR = "https://raw.githubusercontent.com/mwitiderrick/predictive-maintenance/master/ai4i2020.csv"

def download_data(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    target_csv = os.path.join(output_dir, "ai4i2020.csv")

    if os.path.exists(target_csv) and os.path.getsize(target_csv) > 10000:
        print(f"Dataset already exists at {target_csv} ({os.path.getsize(target_csv)} bytes).")
        return target_csv

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    # Attempt 1: Direct UCI Zip archive
    print(f"Attempting download from UCI Repository: {UCI_ZIP_URL}")
    try:
        resp = requests.get(UCI_ZIP_URL, headers=headers, timeout=30)
        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                if csv_files:
                    extracted_name = csv_files[0]
                    with z.open(extracted_name) as f_in, open(target_csv, "wb") as f_out:
                        f_out.write(f_in.read())
                    print(f"Successfully extracted {extracted_name} to {target_csv}")
                    return target_csv
    except Exception as e:
        print(f"UCI zip download failed or timed out: {e}")

    # Attempt 2: Canonical mirrors containing the exact untouched UCI AI4I 2020 CSV
    mirrors = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv",
        GITHUB_RAW_MIRROR,
        KAGGLE_MIRROR,
    ]

    for mirror_url in mirrors:
        print(f"Attempting download from mirror: {mirror_url}")
        try:
            resp = requests.get(mirror_url, headers=headers, timeout=20)
            if resp.status_code == 200 and len(resp.content) > 10000:
                with open(target_csv, "wb") as f:
                    f.write(resp.content)
                print(f"Successfully downloaded dataset from {mirror_url} to {target_csv}")
                return target_csv
        except Exception as e:
            print(f"Mirror {mirror_url} failed: {e}")

    print(f"Automated download failed. Please manually download ai4i2020.csv from {UCI_ZIP_URL} and place it in {output_dir}/ai4i2020.csv")
    return None

if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    res = download_data(out_dir)
    if res and os.path.exists(res):
        df = pd.read_csv(res)
        print(f"Loaded CSV successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
        print(f"Columns: {list(df.columns)}")
        print("Data sample:")
        print(df.head(2))
    else:
        sys.exit(1)
