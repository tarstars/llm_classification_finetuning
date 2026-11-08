"""Derive the fixed followup from the unchanged native engineering notebook."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256
from arena.notebook_worker import worker_cell


def main():
    root=ROOT/'runs/span_019'; pub=REPO/'results/span_019'; role='classifier_followup'
    assert not (root/'jobs'/role/'push_intent.json').exists(),'Preserve accepted source'
    base=read_json(pub/'classifier_pilot_bundle.json')
    original=root/'notebooks/classifier_pilot/job.ipynb'
    assert sha256(original)==base['notebook_sha256']
    preparation=read_json(pub/'quality_preparation.json')
    assert preparation['complete'] and not preparation['uses_winner_labels']
    inputs=read_json(REPO/'results/span_016/matched_input_manifest.json')
    source=next(e for e in inputs['entries'] if e['filename']=='source.parquet')
    notebook=nbf.read(original,4); setup=notebook.cells[1].source
    assert setup.count('32400 if IS_RERUN else 3600')==1
    setup=setup.replace('32400 if IS_RERUN else 3600','32400 if IS_RERUN else 14400')
    insertion=f"(PACKAGE/'span019_followup.py').write_text({(REPO/'src/arena/span019_followup.py').read_text()!r})\n"
    insertion+=f"(PACKAGE/'span019_checked_infer.py').write_text({(REPO/'src/arena/span019_checked_infer.py').read_text()!r})\n"
    insertion+=f"(WORK/'quality_routes.json').write_text({json.dumps(preparation)!r})\n"
    insertion+=f'''SOURCE=None
if not IS_RERUN:
 MATCHED_INPUT=next(p for p in [Path('/kaggle/input/llm-span016-lengthmatched-inputs'),Path('/kaggle/input/datasets/arseniytrushin/llm-span016-lengthmatched-inputs')] if p.exists())
 SOURCE=MATCHED_INPUT/'source.parquet'
 assert SOURCE.stat().st_size=={source['bytes']} and digest(SOURCE)=={source['sha256']!r},'Pinned source texts differ'
 (WORK/'quality_input_verification.json').write_text(json.dumps(dict(passed=True,source_sha256={source['sha256']!r},labels_read=False)))
'''
    assert setup.count('os.chdir(WORK)')==1
    setup=setup.replace('os.chdir(WORK)',insertion+'os.chdir(WORK)')
    old=next(line for line in setup.splitlines() if line.startswith('command='))
    new=old.replace("'arena.span019_pipeline'","'arena.span019_followup'")
    new=new.replace("['--qualify','--engineering-only']",
        "['--qualify','--source',str(SOURCE),'--source-sha256',"+repr(source['sha256'])+",'--routes',str(WORK/'quality_routes.json')]")
    assert new!=old and '--engineering-only' not in new
    setup=setup.replace(old,new)
    actual=worker_cell(role,'')
    notebook.cells=[nbf.v4.new_markdown_cell(
        '# H65 fixed classifier comparison\n\nThe unchanged native replicas and retained Gemma form one equal-weight candidate. Public mode completes fixed engineering and comparison inputs; human votes are evaluated locally only after numerical/runtime checks. Hidden mode reads actual test text and pinned model assets.'),
        nbf.v4.new_code_cell(setup),nbf.v4.new_code_cell(actual)]
    for cell in notebook.cells:
        if cell.cell_type=='code':compile(cell.source,'span019-followup-serialized','exec')
    folder=root/'notebooks'/role;folder.mkdir(parents=True,exist_ok=True)
    nbf.write(notebook,folder/'job.ipynb')
    metadata=read_json(root/'notebooks/classifier_pilot/kernel-metadata.json')
    metadata.update(id='arseniytrushin/llm-preference-span019-classifier-followup',
        title='LLM Preference Span019 Fixed Classifier Comparison')
    metadata['dataset_sources'].append(inputs['dataset'])
    atomic_json(folder/'kernel-metadata.json',metadata)
    checks=[]
    for name,program,limit,success in [('success','print(123)',10,True),
        ('failure','raise SystemExit(7)',10,False),('watchdog','import time;time.sleep(10)',.2,False)]:
        with tempfile.TemporaryDirectory(prefix='span019-followup-') as tmp:
            scope=dict(WORK=Path(tmp),STARTED=time.time(),DEADLINE=time.time()+limit,
                command=[sys.executable,'-c',program],env=None)
            failed=False
            try:exec(compile(actual,'actual-followup-worker','exec'),scope)
            except AssertionError:failed=True
            result=read_json(Path(tmp)/'session_summary.json')
            assert failed!=success and (result['exit_code']==0)==success
            checks.append(dict(name=name,passed=True,summary=result))
    mode='\n'.join(line for line in setup.splitlines() if line.startswith(('IS_RERUN=','DEADLINE=','command=')))
    modes=[]
    for flag,observed,hidden in [('',base['visible_test_sha256'],False),('','changed',True),('true',base['visible_test_sha256'],True)]:
        scope=dict(os=SimpleNamespace(environ={'KAGGLE_IS_COMPETITION_RERUN':flag}),
            hashlib=SimpleNamespace(sha256=lambda data:SimpleNamespace(hexdigest=lambda:observed)),
            RAW=ROOT/'raw',WORK=Path('/tmp'),ASSETS=Path('/tmp/assets'),ORIGIN=Path('/tmp/origin'),
            SOURCE=Path('/tmp/source.parquet'),sys=sys,STARTED=0)
        exec(compile(mode,'actual-followup-branch','exec'),scope)
        assert scope['IS_RERUN']==hidden and ('--qualify' in scope['command'])!=hidden
        assert ('--source' in scope['command'])!=hidden
        assert scope['DEADLINE']==(32400 if hidden else 14400)-120
        modes.append(dict(hidden=hidden,passed=True))
    tiny=nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell('from pathlib import Path\nimport sys,time,tempfile\n'
        'tmp=tempfile.TemporaryDirectory();WORK=Path(tmp.name);STARTED=time.time();DEADLINE=STARTED+10\n'
        'command=[sys.executable,"-c","print(123)"];env=None'),nbf.v4.new_code_cell(actual)])
    executed=NotebookClient(tiny,timeout=30,kernel_name='python3').execute()
    assert all(c.execution_count is not None for c in executed.cells)
    test_output=[]
    for pattern in ['test_span019_followup.py','test_span019_checked_infer.py']:
        tested=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests',
            '-p',pattern,'-v'],cwd=REPO,env={**os.environ,'PYTHONPATH':str(REPO/'src')},
            capture_output=True,text=True)
        assert tested.returncode==0 and 'skipped' not in tested.stderr,tested.stdout+tested.stderr
        test_output.append(tested.stdout+tested.stderr)
    atomic_json(pub/(role+'_cell_rehearsals.json'),dict(passed=True,worker=checks,mode_checks=modes,
        notebook_startup_passed=True,actual_pipeline_cpu_tests_passed=True,
        test_output='\n'.join(test_output),real_model_gpu_exercised=False,new_gpu_executions=0))
    sources=dict(base['source_sha256'])
    for name in ['scripts/build_span_019_followup.py','scripts/run_span_019_followup.py',
        'src/arena/span019_followup.py','tests/test_span019_followup.py','src/arena/span019_quality.py',
        'scripts/audit_span_019_followup.py','src/arena/span019_checked_infer.py',
        'tests/test_span019_checked_infer.py',
        'tests/test_span019_quality.py','scripts/prepare_span_019_quality.py','docs/SPAN_019_QUALITY_PROTOCOL.md',
        'results/span_019/quality_preparation.json','results/span_016/matched_input_manifest.json']:
        sources[name]=sha256(REPO/name)
    atomic_json(pub/(role+'_bundle.json'),dict(role=role,source_sha256=sources,
        notebook_sha256=sha256(folder/'job.ipynb'),metadata_sha256=sha256(folder/'kernel-metadata.json'),
        public_cap_seconds=14400,hidden_cap_seconds=32400,public_stage='fixed_quality_inputs',
        official_submission_eligible=False,internet_enabled=False,
        visible_test_sha256=base['visible_test_sha256'],asset_manifest_sha256=base['asset_manifest_sha256'],
        quality_preparation_sha256=sha256(pub/'quality_preparation.json'),
        source_text_sha256=source['sha256']))
    print(folder)


if __name__=='__main__':main()
