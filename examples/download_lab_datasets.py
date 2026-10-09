"""Optional: download the two CSVs used in the original IBM lab into data/lab/ (free, no API).

    python examples/download_lab_datasets.py
    data-agent "Train a model on each dataset" --data-dir data/lab --provider offline
"""

import urllib.request
from pathlib import Path

URLS = {
    "regression-dataset.csv": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/N0CceRlquaf9q85PK759WQ/regression-dataset.csv",
    "classification-dataset.csv": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/7J73m6Nsz-vmojwab91gMA/classification-dataset.csv",
}
target = Path(__file__).resolve().parents[1] / "data" / "lab"
target.mkdir(parents=True, exist_ok=True)
for name, url in URLS.items():
    path = target / name
    if path.exists():
        print(f"✔ {path} already present")
    else:
        urllib.request.urlretrieve(url, path)
        print(f"⬇ {path}")
