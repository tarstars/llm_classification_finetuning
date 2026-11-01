"""Download original competition files; credentials stay in Kaggle's normal store."""
from pathlib import Path
import base64
import json
import shutil
import urllib.request
import zipfile

ROOT = Path.home() / "database/llm_classification_finetuning"


def main():
    raw = ROOT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    token = Path.home() / ".kaggle/access_token"
    if token.exists():
        authorization = "Bearer " + token.read_text().strip()
    else:
        credentials = json.loads((Path.home() / ".kaggle/kaggle.json").read_text())
        authorization = "Basic " + base64.b64encode(f"{credentials['username']}:{credentials['key']}".encode()).decode()
    headers = {"Authorization": authorization}
    archive_path = raw / "llm-classification-finetuning.zip"
    url = "https://www.kaggle.com/api/v1/competitions/data/download-all/llm-classification-finetuning"
    if not archive_path.exists():
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180) as src, archive_path.with_suffix(".part").open("wb") as dst:
            shutil.copyfileobj(src, dst)
        archive_path.with_suffix(".part").replace(archive_path)
    with zipfile.ZipFile(raw / "llm-classification-finetuning.zip") as archive:
        for name in ("train.csv", "test.csv", "sample_submission.csv"):
            with archive.open(name) as src, (raw / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
    print("Downloaded original files to", raw)


if __name__ == "__main__":
    main()
