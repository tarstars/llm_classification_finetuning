"""Verify a portable native CUDA artifact offline against the current judge cache."""
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zlib

import nbformat as nbf
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from arena.data import ROOT, json_write

root = ROOT / "runs/span_003"
dest = root / "notebooks/native_profile"
dest.mkdir(parents=True, exist_ok=True)
assert not (root / "jobs/native_profile/push_intent.json").exists()
selection = json.loads((ROOT / "runs/attempt_003/judge_selection.json").read_text())
for name, key in [("judge", "judge_source_sha256"), ("judge_runtime", "runtime_source_sha256")]:
    assert hashlib.sha256((REPO / f"src/arena/{name}.py").read_bytes()).hexdigest() == selection[key]
sample = {"ids": json.loads((root / "sample_ids.json").read_text())["native_profile"]}
inventory = json.loads((REPO / "results/span_003/artifact_inventory.json").read_text())
caches = [np.load(inventory[name]["path"]) for name in ["judge_fit", "judge_extra"]]
ids = np.concatenate([data["ids"] for data in caches])
predictions = np.concatenate([data["original"] for data in caches])
lookup = {int(rid): index for index, rid in enumerate(ids)}
reference = predictions[[lookup[int(rid)] for rid in sample["ids"]]]
stream = io.BytesIO()
np.savez_compressed(stream, ids=np.array(sample["ids"]), original=reference)
blob = base64.b64encode(zlib.compress(stream.getvalue())).decode()
license_path = root / "license_files/llama.cpp-LICENSE"
license_text = license_path.read_text()
old = ROOT / "runs/span_002/notebooks/ensemble-extra/job.ipynb"
nb = nbf.read(old, as_version=4)
nb.cells = nb.cells[:5]
nb.cells[0].source = "# Span003 E02 — portable native CUDA artifact, Internet disabled\n\nRetain the existing judge's model, prompt, packing, serving settings and raw-class mapping."
nb.cells[1].source = re.sub(r"^SAMPLE=.*$", "SAMPLE=" + repr(sample), nb.cells[1].source, flags=re.MULTILINE)
nb.cells[1].source = nb.cells[1].source.replace("BIN=WORK/'runtime_bin';", "BUNDLE=WORK/'offline_native_runtime';BUNDLE.mkdir(exist_ok=True)\nBIN=BUNDLE/'bin';")
nb.cells[1].source = nb.cells[1].source.replace("SESSION_STARTED=time.time()", """SESSION_STARTED=time.time()
import signal
DEADLINE=SESSION_STARTED+1800-60
def expired(*unused):raise TimeoutError('Native profile deadline reached')
signal.signal(signal.SIGALRM,expired);signal.alarm(max(1,int(DEADLINE-time.time())))""")
nb.cells.append(nbf.v4.new_code_cell(f'''import subprocess,importlib.metadata,hashlib,zlib
from arena import judge_runtime
REPAIR=next(p for p in [Path('/kaggle/input/llm-preference-span-002-judge-recovery'),
    Path('/kaggle/input/notebooks/arseniytrushin/llm-preference-span-002-judge-recovery')]
    if (p/'cuda12').is_dir())
shutil.copytree(REPAIR/'cuda12',BUNDLE/'cuda12',dirs_exist_ok=True)
shutil.copy2(RUNTIME/'runtime/runtime.json',BUNDLE/'runtime.json')
(BUNDLE/'licenses').mkdir(exist_ok=True)
(BUNDLE/'licenses/llama.cpp-LICENSE').write_text({license_text!r})
packages={{d.metadata['Name']:d.version for d in importlib.metadata.distributions(path=[str(BUNDLE/'cuda12')])}}
assert packages['nvidia-cublas-cu12']=='12.8.4.1'
assert packages['nvidia-cuda-runtime-cu12']=='12.8.90'
license_files=[p for p in (BUNDLE/'cuda12').rglob('*') if p.is_file() and ('license' in p.name.lower() or 'license' in str(p.parent).lower())]
assert len(license_files)>=2,'Retain NVIDIA license notices in the portable artifact'
paths=[str(BUNDLE/'cuda12/nvidia/cublas/lib'),str(BUNDLE/'cuda12/nvidia/cuda_runtime/lib')]
for path in [Path(paths[0])/'libcublas.so.12',Path(paths[0])/'libcublasLt.so.12',Path(paths[1])/'libcudart.so.12']:
    assert path.is_file() and path.stat().st_size>1_000_000,'Missing/corrupt saved CUDA library'
original_environment=judge_runtime.runtime_environment
def offline_environment(executable):
    result=original_environment(executable)
    result['LD_LIBRARY_PATH']=':'.join(paths)+':'+result.get('LD_LIBRARY_PATH','')
    return result
judge_runtime.runtime_environment=offline_environment
environment=judge_runtime.runtime_environment(EXE)
backend=next(BIN.glob('libggml-cuda.so*'))
dependency=subprocess.run(['ldd',str(backend)],env=environment,capture_output=True,text=True,timeout=30)
assert dependency.returncode==0 and 'not found' not in dependency.stdout
devices=subprocess.run([str(EXE),'--list-devices'],env=environment,capture_output=True,text=True,timeout=60)
device_text=devices.stdout+devices.stderr
assert devices.returncode==0 and 'CUDA0' in device_text and 'CUDA1' in device_text
files={{}}
for path in sorted(BUNDLE.rglob('*')):
    if path.is_file():
        with path.open('rb') as handle:sha=hashlib.file_digest(handle,'sha256').hexdigest()
        files[str(path.relative_to(BUNDLE))]={{'sha256':sha,'bytes':path.stat().st_size}}
manifest={{'packages':packages,'files':files,'original_runtime':CONFIG['llama_revision'],
    'sources':{{'binary':{selection['runtime_kernel']!r},'cuda':'arseniytrushin/llm-preference-span-002-judge-recovery/2'}},
    'license_source':'https://raw.githubusercontent.com/ggml-org/llama.cpp/{selection['config']['llama_revision']}/LICENSE'}}
(BUNDLE/'manifest.json').write_text(json.dumps(manifest,indent=2))
(WORK/'hardware.json').write_text(json.dumps({{'ldd':dependency.stdout,'devices':device_text,
    'portable_manifest_sha256':hashlib.sha256((BUNDLE/'manifest.json').read_bytes()).hexdigest(),
    'library_files':len(files),'packages':packages}},indent=2))
(WORK/'native_reference.npz').write_bytes(zlib.decompress(__import__('base64').b64decode({blob!r})))
'''))
nb.cells.append(nbf.v4.new_code_cell('''import numpy as np,threading
from arena.judge import run
stop=threading.Event();memory_samples=[]
def memory_worker():
    while not stop.is_set():
        try:
            values=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5)
            memory_samples.append([int(value.strip()) for value in values.splitlines()])
        except (subprocess.SubprocessError,ValueError):pass
        stop.wait(.5)
worker=threading.Thread(target=memory_worker,daemon=True);worker.start()
profile_started=time.time()
try:summary=run(CONFIG,EXE,RAW,WORK/'native','gemma26','dev',sample=SAMPLE,both_orders=False)
finally:stop.set();worker.join(timeout=6)
observed=np.load(WORK/'native/judge_predictions.npz')
reference=np.load(WORK/'native_reference.npz')
np.testing.assert_array_equal(observed['ids'],reference['ids'])
assert len(observed['ids'])==64
delta=np.abs(observed['original']-reference['original'])
assert memory_samples and all(len(row)==2 for row in memory_samples)
peak=np.max(np.asarray(memory_samples),axis=0)
assert np.all(peak>1024),'Actual GPU model allocation not established on both devices'
gate={'rows':64,'mean_probability_difference':float(delta.mean()),'max_probability_difference':float(delta.max()),
      'mean_limit':.01,'max_limit':.1,'identity_passed':bool(delta.mean()<=.01 and delta.max()<=.1),
      'seconds_per_row':summary['seconds']/64,'scoring_seconds':summary['seconds'],
      'setup_seconds_before_profile':profile_started-SESSION_STARTED,
      'sampled_gpu_used_peak_mib':peak.tolist(),'memory_samples':len(memory_samples),
      'memory_note':'Sampled total device memory used, including driver/runtime; not allocator peak memory.',
      'native_both_devices_verified':True,'session_seconds':time.time()-SESSION_STARTED}
(WORK/'native_gate.json').write_text(json.dumps(gate,indent=2))
(WORK/'session_summary.json').write_text(json.dumps({'session_seconds':time.time()-SESSION_STARTED,
    'native_profile_complete':True,'identity_passed':gate['identity_passed']},indent=2))
print('NATIVE_GATE',json.dumps(gate),flush=True)
assert gate['identity_passed'],'Preserve failed guard; do not expand or silently relax limits'
signal.alarm(0)
'''))
for cell in nb.cells:
    if cell.cell_type == "code":
        compile(cell.source, "native-offline-bundle", "exec")
nbf.write(nb, dest / "job.ipynb")
metadata = json.loads((ROOT / "runs/span_002/notebooks/ensemble-extra/kernel-metadata.json").read_text())
metadata.update(id="arseniytrushin/llm-preference-span-003-native-profile", title="LLM Preference Span 003 Native Profile", enable_internet=False)
metadata["kernel_sources"] = [selection["runtime_kernel"], "arseniytrushin/llm-preference-span-002-judge-recovery/2"]
json_write(dest / "kernel-metadata.json", metadata)
json_write(REPO / "results/span_003/native_profile_bundle.json", {
    "notebook_sha256": hashlib.sha256((dest / "job.ipynb").read_bytes()).hexdigest(),
    "parent_source_notebook_sha256": hashlib.sha256(old.read_bytes()).hexdigest(),
    "maximum_session_seconds": 1800, "profile_rows": 64, "internet_enabled": False,
    "native_source_sha256": selection["runtime_source_sha256"], "judge_source_sha256": selection["judge_source_sha256"],
    "license_sha256": hashlib.sha256(license_path.read_bytes()).hexdigest(),
    "source_revision": selection["config"]["llama_revision"], "parent_kernels": metadata["kernel_sources"]})
print(dest)
