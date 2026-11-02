"""Manage an explicitly named Kaggle job without modifying attempt 001."""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from arena.data import ROOT, json_write

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["push", "status", "output", "submit", "submissions", "quota"])
parser.add_argument("--folder", type=Path, required=True)
parser.add_argument("--job", required=True)
parser.add_argument("--pattern", default=r"(?:summary|gpu_probe|training_config|confirmation|prediction_summary|runtime_requirements)\.json$|training_progress\.jsonl$|\.csv$|\.log$|\.ipynb$")
parser.add_argument("--accelerator")
parser.add_argument("--timeout", type=int, default=32400)
args = parser.parse_args()
folder = args.folder.resolve()
metadata = json.loads((folder / "kernel-metadata.json").read_text())
out = ROOT / "runs/attempt_002/jobs" / args.job
out.mkdir(parents=True, exist_ok=True)
token = Path.home() / ".kaggle/access_token"
if token.exists():
    os.environ["KAGGLE_API_TOKEN"] = token.read_text().strip()
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
reference = metadata['id']
saved_push = json.loads((out/'push.json').read_text()) if (out/'push.json').exists() else None
if saved_push and saved_push.get('ref','').startswith('/code/'):
    reference=saved_push['ref'].removeprefix('/code/')
submission_kernel=reference
if args.action in ['status', 'output'] and saved_push:
    reference += '/' + str(saved_push['version_number'])


def fields(obj):
    names = ["url", "version_number", "error", "message", "status", "ref", "id",
             "failure_message", "invalid_model_sources", "invalid_dataset_sources",
             "invalidModelSources", "invalidDatasetSources", "public_score", "private_score", "description", "date"]
    return {k: getattr(obj, k) for k in names if hasattr(obj, k)}


if args.action == "push":
    result = fields(api.kernels_push(str(folder), timeout=str(args.timeout), acc=args.accelerator))
    json_write(out / "push.json", result)
    print(json.dumps(result, default=str, indent=2))
    if result.get("error"):
        raise SystemExit(1)
elif args.action == "status":
    result = fields(api.kernels_status(reference))
    json_write(out / "status.json", result)
    print(json.dumps(result, default=str, indent=2))
elif args.action == "output":
    paths, status = api.kernels_output(reference, path=str(out / "output"),
        file_pattern=args.pattern, force=True, quiet=True, page_size=100)
    print(json.dumps({"files": paths, "status": status}, indent=2))
elif args.action == "submit":
    version = json.loads((out / "push.json").read_text())["version_number"]
    result = fields(api.competition_submit_code(file_name="submission.csv",
        message=f"Attempt 002: joint preference classifier ({args.job})",
        competition="llm-classification-finetuning", kernel=submission_kernel, kernel_version=version))
    json_write(out / "submission.json", result)
    print(json.dumps(result, default=str, indent=2))
elif args.action == 'submissions':
    result = [fields(x) for x in api.competition_submissions('llm-classification-finetuning') or []]
    json_write(out/'submissions.json',result)
    print(json.dumps(result,default=str,indent=2))
elif args.action == 'quota':
    quota=api.quota_view()
    gpu=quota.gpu_quota
    result={'refresh_time':str(quota.quota_refresh_time),
            'used_seconds':gpu.time_used.total_seconds(),
            'reserved_seconds':gpu.time_reserved.total_seconds(),
            'allowed_seconds':gpu.total_time_allowed.total_seconds(),
            'pay_to_scale_enabled':gpu.is_pay_to_scale_enabled}
    result['remaining_seconds']=result['allowed_seconds']-result['used_seconds']-result['reserved_seconds']
    json_write(out/'quota.json',result)
    print(json.dumps(result,indent=2))
