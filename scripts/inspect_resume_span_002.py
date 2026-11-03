"""Inspect real immutable output paths and fetch only checkpoint state metadata."""
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, json_write

token = Path.home() / '.kaggle/access_token'
if token.exists():
    os.environ['KAGGLE_API_TOKEN'] = token.read_text().strip()
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
api = KaggleApi()
api.authenticate()
files, page = [], None
while True:
    with api.build_kaggle_client() as client:
        request = ApiListKernelSessionOutputRequest()
        request.user_name = 'arseniytrushin'
        request.kernel_slug = 'llm-preference-qwen3-4b-train-002'
        request.version_label = api._kernel_version_label('1')
        api._set_paging(request, 100, page)
        result = client.kernels.kernels_api_client.list_kernel_session_output(request)
    files.extend(item.file_name for item in result.files)
    page = result.next_page_token
    if not page:
        break
root = ROOT / 'runs/span_002'
kernel = 'arseniytrushin/llm-preference-qwen3-4b-train-002/1'
paths, _ = api.kernels_output(kernel, path=str(root / 'baseline_resume'),
    file_pattern=r'checkpoints/.*/(?:trainer_state\.json|adapter_config\.json)$',
    force=False, quiet=True, page_size=100)
report = {'kernel': kernel, 'file_paths': files, 'downloaded_metadata': paths,
    'note': 'Kernels list-files returns basenames; session-output listing gives checkpoint-relative paths.'}
json_write(root / 'resume_inventory.json', report)
json_write(REPO / 'results/span_002/resume_inventory.json', report)
print(json.dumps({'checkpoint_files': [x for x in files if '/checkpoints/' in x],
                  'downloaded_metadata': paths}, indent=2))
