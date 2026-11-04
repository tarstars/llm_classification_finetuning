"""Bundle frozen H03 export verification or official prediction, without training."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import zlib

import nbformat as nbf
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from arena.data import ROOT, json_write
from arena.span_heads import temperature

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("role", choices=["classifier_smoke", "classifier_predict"])
args = parser.parse_args()
root = ROOT / "runs/span_003"
dest = root / "notebooks" / args.role
dest.mkdir(parents=True, exist_ok=True)
assert not (root / "jobs" / args.role / "push_intent.json").exists(), "Never overwrite launched source"
manifest = json.loads((REPO / "results/span_002/research_candidate.json").read_text())
parent = json.loads((ROOT / "runs/attempt_002/model_selection.json").read_text())
config = json.loads((REPO / "configs/span_003.json").read_text())
cap = config["job_seconds_caps"][args.role]
adapter = ROOT / "runs/span_002/jobs/exposure/output/exposure/adapter"
manifest["tokenizer_sha256"] = hashlib.sha256((adapter / "tokenizer.json").read_bytes()).hexdigest()
profile_rate = 0.
if args.role == "classifier_predict":
    smoke = json.loads((root / "jobs/classifier_smoke/output/smoke/smoke_summary.json").read_text())
    assert smoke["passed"] and smoke["adapter_sha256"] == manifest["adapter_sha256"]
    previous = json.loads((REPO / "results/span_002/exposure_results.json").read_text())
    profile_rate = max(smoke["seconds_per_row"], previous["summary"]["development_prediction_seconds"] / 5746)
owner, training_slug = parent["kernel"].split("/")
export_owner, export_slug = manifest["source_kernel"].split("/")
source = manifest["config"]["model_source"]
cells = [nbf.v4.new_markdown_cell("# Span003 — frozen classifier " + args.role),
    nbf.v4.new_code_cell(f'''import os,sys,time,json,subprocess,base64,zlib,hashlib
from pathlib import Path
SESSION_STARTED=time.time()
DEADLINE=SESSION_STARTED+{cap}-60
WORK=Path('/kaggle/working')
os.environ.update(TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
TRAIN=next(p for p in [Path('/kaggle/input/{training_slug}'),Path('/kaggle/input/notebooks/{owner}/{training_slug}')]
           if (p/'runtime_requirements.json').exists())
EXPORT=next(p for p in [Path('/kaggle/input/{export_slug}'),Path('/kaggle/input/notebooks/{export_owner}/{export_slug}')]
            if (p/'exposure/adapter/adapter_model.safetensors').exists())
runtime=json.loads((TRAIN/'runtime_requirements.json').read_text())
subprocess.run([sys.executable,'-m','pip','install','-q','--no-index','--no-deps',
    '--find-links',str(TRAIN/'runtime_wheels'),*runtime],check=True,timeout=240)
MODULES=WORK/'bundled_code';(MODULES/'arena').mkdir(parents=True,exist_ok=True)
(MODULES/'arena/__init__.py').write_text('')
MANIFEST={manifest!r}
(WORK/'frozen_manifest.json').write_text(json.dumps(MANIFEST,indent=2))
''')]
sources = {}
for name in ["data", "joint_data", "joint_train", "joint_infer", "span_export"]:
    path = REPO / f"src/arena/{name}.py"
    sources[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    cells.append(nbf.v4.new_code_cell(f"(MODULES/'arena/{name}.py').write_text({path.read_text()!r})"))
if args.role == "classifier_smoke":
    samples = json.loads((root / "sample_ids.json").read_text())["classifier_smoke"]
    cache = np.load(ROOT / "runs/span_002/jobs/exposure/output/exposure/dev_predictions.npz")
    lookup = {int(rid): index for index, rid in enumerate(cache["ids"])}
    idx = [lookup[int(rid)] for rid in samples]
    original, aligned = cache["original"][idx], cache["aligned_swap"][idx]
    stream = io.BytesIO()
    np.savez_compressed(stream, ids=np.array(samples), original=original, aligned_swap=aligned,
                        average=(original+aligned)/2, calibrated=temperature((original+aligned)/2, manifest["temperature"]))
    blob = base64.b64encode(zlib.compress(stream.getvalue())).decode()
    cells.append(nbf.v4.new_code_cell(f"(WORK/'reference.npz').write_bytes(zlib.decompress(base64.b64decode({blob!r})))"))
tail = "['--mode','smoke','--reference',str(WORK/'reference.npz')]" if args.role == "classifier_smoke" else f"['--mode','predict','--profile-rate',{str(profile_rate)!r}]"
cells.append(nbf.v4.new_code_cell(f'''import torch
assert torch.cuda.device_count()==2
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),
                    Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'test.csv').exists())
MODEL=next(p for p in [Path('/kaggle/input/{'/'.join(source.split('/')[1:])}'),
                      Path('/kaggle/input/models/{source}'),TRAIN/'base_model'] if (p/'config.json').exists())
environment=os.environ.copy();environment.update(PYTHONPATH=str(MODULES),PYTHONUNBUFFERED='1')
def execute(mode_arguments,out):
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        '--module','arena.span_export','--raw',str(RAW),'--model',str(MODEL),
        '--adapter',str(EXPORT/'exposure/adapter'),'--manifest',str(WORK/'frozen_manifest.json'),
        '--out',str(out),'--deadline',str(DEADLINE),*mode_arguments]
    with (WORK/('console_'+out.name+'.log')).open('w') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=environment)
        try:
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            status=process.wait(timeout=max(1,DEADLINE-time.time()))
        finally:
            if process.poll() is None:process.terminate()
        assert status==0,'Frozen inference failed; retain exact source/logs and do not submit'
execute({tail},WORK/{'smoke' if args.role=='classifier_smoke' else 'prediction'!r})
'''))
if args.role == "classifier_smoke":
    cells.append(nbf.v4.new_code_cell('''summary=json.loads((WORK/'smoke/smoke_summary.json').read_text())
assert summary['passed']
rate=summary['seconds_per_row']
execute(['--mode','predict','--profile-rate',str(rate)],WORK/'visible')
visible=json.loads((WORK/'visible/prediction_summary.json').read_text())
assert visible['probabilities_valid']
(WORK/'session_summary.json').write_text(json.dumps({'seconds':time.time()-SESSION_STARTED,
    'smoke_passed':True,'visible_rows':visible['rows'],'visible_probabilities_valid':True},indent=2))
'''))
else:
    cells.append(nbf.v4.new_code_cell('''import shutil
shutil.copyfile(WORK/'prediction/submission.csv',WORK/'submission.csv')
(WORK/'session_summary.json').write_text(json.dumps({'seconds':time.time()-SESSION_STARTED,
    'prediction_complete':True},indent=2))
'''))
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
for cell in nb.cells:
    if cell.cell_type == "code":
        compile(cell.source, "classifier-bundle", "exec")
nbf.write(nb, dest / "job.ipynb")
metadata = {"id": "arseniytrushin/llm-preference-span-003-" + args.role.replace("_", "-"),
    "title": "LLM Preference Span 003 " + args.role.replace("_", " ").title(),
    "code_file": "job.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_internet": False, "competition_sources": ["llm-classification-finetuning"],
    "dataset_sources": [], "kernel_sources": [parent["kernel"] + "/" + str(parent["version_number"]),
        manifest["source_kernel"] + "/" + str(manifest["version_number"])], "model_sources": [source]}
json_write(dest / "kernel-metadata.json", metadata)
json_write(REPO / f"results/span_003/{args.role}_bundle.json", {
    "notebook_sha256": hashlib.sha256((dest / "job.ipynb").read_bytes()).hexdigest(),
    "source_sha256": sources, "maximum_session_seconds": cap, "profile_rate": profile_rate,
    "adapter_sha256": manifest["adapter_sha256"], "tokenizer_sha256": manifest["tokenizer_sha256"],
    "parent_kernels": metadata["kernel_sources"], "internet_enabled": False})
print(dest)
