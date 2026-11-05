"""One fenced v6 GPU mutation; exact-version observation and bounded artifact reads."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import argparse,fcntl,hashlib,json,os,re,subprocess,sys,time
import requests
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write
from arena.span004_support import api_client,inventory,quota,read_call,TERMINAL,public_inventory
from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def retrieve(api,exact,directory):
    owner,slug,version=api.parse_kernel_string(exact);cursor=None;files=[]
    for _ in range(20):
        def listing():
            req=ApiListKernelSessionOutputRequest();req.user_name=owner;req.kernel_slug=slug;req.version_label=api._kernel_version_label(version);api._set_paging(req,100,cursor)
            with api.build_kaggle_client() as client:return client.kernels.kernels_api_client.list_kernel_session_output(req)
        response=read_call(listing);files.extend(response.files);cursor=response.next_page_token
        if not cursor:break
    assert not cursor
    selected=[item for item in files if re.search(r'\.(json|npz|log|csv)$',item.file_name) and not item.file_name.startswith('bundled_code/')]
    output=directory/'output';output.mkdir(exist_ok=True)
    def fetch(item):
        target=output/item.file_name;assert target.resolve().is_relative_to(output.resolve());target.parent.mkdir(parents=True,exist_ok=True)
        def get():
            r=requests.get(item.url,headers={'Accept-Encoding':'identity'},timeout=(10,45));r.raise_for_status();return r
        r=read_call(get);data=r.content
        if r.headers.get('Content-Length') and not r.headers.get('Content-Encoding'):assert len(data)==int(r.headers['Content-Length'])
        temp=target.with_suffix(target.suffix+'.download');temp.write_bytes(data);temp.replace(target)
        return item.file_name,sha(target)
    with ThreadPoolExecutor(max_workers=3) as pool:hashes=dict(pool.map(fetch,selected))
    json_write(directory/'retrieval.json',{'files':len(hashes),'artifact_sha256':hashes,'new_gpu_executions':0,'signed_urls_persisted':False})
    return hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('role',choices=['confirmation','offline_profile']);parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    root=ROOT/'runs/span_006';pub=REPO/'results/span_006';directory=root/'jobs'/args.role;public=pub/'jobs'/args.role
    directory.mkdir(parents=True,exist_ok=True);public.mkdir(parents=True,exist_ok=True)
    lock=(directory/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=read(REPO/'configs/span_006.json');plan=read(pub/'span_plan.json');bundle=read(pub/(args.role+'_bundle.json'));folder=root/'notebooks'/args.role;metadata=read(folder/'kernel-metadata.json');cap=cfg['job_seconds_caps'][args.role]
    assert sha(folder/'job.ipynb')==bundle['notebook_sha256'] and sha(folder/'kernel-metadata.json')==bundle['metadata_sha256'] and sha(REPO/'configs/span_006.json')==bundle['config_sha256']
    for path,digest in bundle['source_sha256'].items():
        assert sha(REPO/path)==digest,path
        assert hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+path],cwd=REPO)).hexdigest()==digest,'Commit exact sources before GPU launch'
    assert metadata['enable_internet'] is False and cap<=25200
    deadline=datetime.fromisoformat(plan['deadline']).timestamp()
    def save(name,value):
        for dest in [directory,public]:json_write(dest/name,public_inventory(value) if name=='admission_inventory.json' and dest==public else value)
    def state(stage,**kwargs):
        r={'stage':stage,'at':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'role':args.role,**kwargs};save('controller_state.json',r);print(json.dumps(r),flush=True)
    api=api_client()
    if not (directory/'push.json').exists():
        assert not (directory/'push_intent.json').exists(),'Reconcile ambiguous push; never duplicate'
        admission=(root/'admission.lock').open('a');fcntl.flock(admission,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state('fresh_admission');live=inventory(api,plan['started_at']);q=live['quota'];dependencies=read(pub/'dependency_inventory.json')
        assert not live['nonterminal_jobs'] and q['reserved_seconds']==0 and not q['paid_scaling']
        assert q['refresh_time']==plan['initial_quota_refresh_time'] and q['remaining_seconds']>=cap
        assert 0<=q['used_seconds']-plan['quota_used_at_start']<=plan['gpu_session_seconds_cap']-cap
        assert time.time()+cap+300<deadline and (datetime.now(timezone.utc)-datetime.fromisoformat(dependencies['observed_at'])).total_seconds()<3600
        assert dependencies['pinned_model_version_available'] and set(metadata['kernel_sources'])=={r['source'] for r in dependencies['parents']} and all(r['status']=='COMPLETE' for r in dependencies['parents'])
        if args.role=='offline_profile':assert read(pub/'H46_results.json')['gate']['passed']
        intent={'at':datetime.now(timezone.utc).isoformat(),'kernel':metadata['id'],'quota_before':q,'max_session_seconds':cap,'notebook_sha256':bundle['notebook_sha256'],'config_sha256':bundle['config_sha256']}
        save('admission_inventory.json',live)
        if args.dry_run:state('dry_run_admissible',intent=intent);return
        save('push_intent.json',intent);state('pushing_one_version')
        response=api.kernels_push(str(folder),timeout=str(cap),acc='NvidiaTeslaT4')
        pushed={k:getattr(response,k) for k in ['ref','version_number','url','error','invalid_model_sources','invalid_dataset_sources'] if hasattr(response,k)};save('push.json',pushed)
        assert pushed.get('version_number') and not pushed.get('error') and not pushed.get('invalid_model_sources') and not pushed.get('invalid_dataset_sources')
        fcntl.flock(admission,fcntl.LOCK_UN)
    pushed=read(directory/'push.json');exact=pushed['ref'].removeprefix('/code/')+'/'+str(pushed['version_number']);state('monitoring',kernel_version=exact)
    while time.time()<deadline:
        try:response=read_call(lambda:api.kernels_status(exact))
        except Exception as error:state('observation_retry',error_type=type(error).__name__,kernel_version=exact);time.sleep(30);continue
        status=str(response.status).split('.')[-1];save('status.json',{'status':status,'kernel_version':exact});state('observed',status=status,kernel_version=exact)
        if status in TERMINAL:
            save('quota.json',quota(api));hashes=retrieve(api,exact,directory)
            receipt=read(directory/'retrieval.json');save('retrieval.json',receipt)
            charge=read(directory/'quota.json')['used_seconds']-read(directory/'push_intent.json')['quota_before']['used_seconds']
            state('complete' if status=='COMPLETE' else 'failed',status=status,kernel_version=exact,charged_seconds=charge,files=len(hashes))
            if status!='COMPLETE':raise SystemExit(1)
            return
        time.sleep(45)
    state('original_deadline_reached',kernel_version=exact);raise TimeoutError('Observe/stop exact job; no duplicate execution')


if __name__=='__main__':main()
