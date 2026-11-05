"""Fresh exact runtime/export/model availability; reads only, no GPU launch."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write
from arena.span004_support import api_client,read_call
from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest

api=api_client();sources=['arseniytrushin/llm-preference-qwen3-4b-train-002/1','arseniytrushin/llm-preference-span-002-exposure/1'];parents=[]
for source in sources:
    status=str(read_call(lambda:api.kernels_status(source)).status).split('.')[-1];assert status=='COMPLETE',source
    owner,slug,version=api.parse_kernel_string(source);cursor=None;files=[]
    for _ in range(20):
        def listing():
            request=ApiListKernelSessionOutputRequest();request.user_name=owner;request.kernel_slug=slug;request.version_label=api._kernel_version_label(version);api._set_paging(request,100,cursor)
            with api.build_kaggle_client() as client:return client.kernels.kernels_api_client.list_kernel_session_output(request)
        response=read_call(listing);files.extend(item.file_name for item in response.files);cursor=response.next_page_token
        if not cursor:break
    assert not cursor
    if 'train-002' in source:assert 'runtime_requirements.json' in files and any(x.startswith('runtime_wheels/') for x in files)
    else:assert 'exposure/adapter/adapter_model.safetensors' in files
    parents.append({'source':source,'status':status,'files':files})
model='qwen-lm/qwen-3/transformers/4b';response=read_call(lambda:api.model_instance_versions_list(model,page_size=100))
versions=[v.to_dict() for v in response.version_list.versions];assert not response.next_page_token
assert any(int(v.get('versionNumber',v.get('version_number')))==1 for v in versions)
proof={'observed_at':datetime.now(timezone.utc).isoformat(),'parents':parents,'model_source':model+'/1','pinned_model_version_available':True}
for p in [REPO/'results/span_006',ROOT/'runs/span_006']:json_write(p/'dependency_inventory.json',proof)
print(json.dumps({'observed_at':proof['observed_at'],'parents':[{'source':r['source'],'status':r['status'],'files':len(r['files'])} for r in parents],'model_source':proof['model_source']},indent=2))
