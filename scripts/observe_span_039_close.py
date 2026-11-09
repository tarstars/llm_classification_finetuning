"""Read-only terminal/quota/history/board close for a source-only span."""
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.experiment import atomic_json,read_json,sha256,utc_now
from arena.span004_support import api_client,quota,read_call
from submit_span_013_order import submissions,TERMINAL
from submit_span_010_order import fields


def main():
    pub=REPO/'results/span_039';assert not (pub/'resource_close.json').exists()
    result=read_json(pub/'H81_corrected_results.json');assert result['decision']=='no_predeclared_material_mismatch'
    previous=read_json(REPO/'results/span_038/resource_close.json')
    api=api_client();current=quota(api)
    status=str(read_call(lambda:api.kernels_status('arseniytrushin/llm-preference-span036-pairrm-pilot/1')).status).split('.')[-1]
    history=submissions(api);assert status=='COMPLETE' and len(history)==8
    assert all(str(s.status).split('.')[-1] in TERMINAL for s in history)
    assert current['used_seconds']==previous['quota']['used_seconds']
    assert current['reserved_seconds']==0 and not current['paid_scaling']
    assert current['refresh_time']==previous['quota']['refresh_time']
    atomic_json(pub/'terminal_official_history.json',dict(at=utc_now(),complete=True,count=8,all_terminal=True,new_official_submissions=0,submissions=[fields(s) for s in history]))
    subprocess.run([sys.executable,str(REPO/'scripts/observe_leaderboard.py'),'--output','results/span_039/final_leaderboard.json'],cwd=REPO,check=True,timeout=180)
    record=dict(at=utc_now(),passed=True,quota=current,gpu_charge_seconds=0,public_versions=0,official_submissions=0,
        official_history_complete=True,all_official_terminal=True,official_count=8,last_pilot_status=status,
        full_owned_inventory_refreshed=False,previous_full_inventory_sha256=previous['previous_full_inventory_sha256'],
        inventory_limit='No GPU/job creation in source-only span; full catalog last verified v36. Fresh full admission required before future GPU.',
        leaderboard_sha256=sha256(pub/'final_leaderboard.json'),history_sha256=sha256(pub/'terminal_official_history.json'),
        result_sha256=sha256(pub/'H81_corrected_results.json'),source_sha256=sha256(Path(__file__)))
    atomic_json(pub/'resource_close.json',record);print(record)


if __name__=='__main__':main()
