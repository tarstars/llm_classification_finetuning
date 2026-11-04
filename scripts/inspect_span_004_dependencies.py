"""Verify the exact existing native parents and pinned model remain available."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, json_write
from arena.span004_support import api_client, read_call
from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest

api = api_client()
parents = ['arseniytrushin/llm-preference-judge-binary-probe-003/1',
           'arseniytrushin/llm-preference-span-002-judge-recovery/2']
records = []
for source in parents:
    status = str(read_call(lambda: api.kernels_status(source)).status).split('.')[-1]
    assert status == 'COMPLETE', source
    files, cursor = [], None
    for _ in range(20):
        owner, slug, version = api.parse_kernel_string(source)
        def listing():
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = owner, slug
            request.version_label = api._kernel_version_label(version)
            api._set_paging(request,100,cursor)
            with api.build_kaggle_client() as client:
                return client.kernels.kernels_api_client.list_kernel_session_output(request)
        response = read_call(listing)
        files.extend({'name': item.file_name} for item in response.files)
        cursor = response.next_page_token
        if not cursor:
            break
    assert not cursor
    if 'binary-probe' in source:
        assert any(row['name'].endswith('runtime/bin/llama-server') for row in files)
    else:
        assert all(any(row['name'].endswith(name) for row in files) for name in ['libcudart.so.12', 'libcublas.so.12'])
    records.append({'source': source, 'status': status, 'files': files})
model = 'google/gemma-4/gguf/gemma-4-26b-a4b-it-qat-q4_0-gguf'
versions, cursor = [], None
for _ in range(20):
    response = read_call(lambda: api.model_instance_versions_list(model, page_size=100, page_token=cursor))
    versions.extend(item.to_dict() for item in response.version_list.versions)
    cursor = response.next_page_token
    if not cursor:
        break
assert not cursor
number = lambda row: row.get('versionNumber', row.get('version_number'))
assert any(int(number(row)) == 2 for row in versions), 'Pinned model version unavailable'
result = {'observed_at': datetime.now(timezone.utc).isoformat(), 'parents': records,
          'model_source': model + '/2', 'pinned_model_version_available': True,
          'model_version_metadata': {key:value for key,value in next(row for row in versions if int(number(row)) == 2).items()
                                    if key in {'versionNumber', 'version_number', 'id', 'versionId', 'createdAt', 'totalUncompressedBytes'}}}
for folder in [ROOT / 'runs/span_004', REPO / 'results/span_004']:
    json_write(folder / 'dependency_inventory.json', result)
print(json.dumps({'observed_at': result['observed_at'], 'parents': [{'source':r['source'], 'status':r['status'],
    'files':len(r['files'])} for r in records], 'model_source':result['model_source'],
    'pinned_model_version_available':True},indent=2))
