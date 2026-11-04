"""Record the actual v4 stop/deferral decision without fitting or inference."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write

def main():
    folder=REPO/'results/span_004';read=lambda name:json.loads((folder/name).read_text())
    result,audit=read('repeatability_results.json'),read('terminal_audit.json')
    assert audit['passed'] and result['selected_recipe'] is None
    assert not any(case['qualifies_for_consistent_fitting_cache'] for case in result['cases'].values())
    assert audit['fresh_requests_reconstructed']==512 and not audit['new_official_submissions']
    value={'span':4,'generated_at':datetime.now(timezone.utc).isoformat(),
        'E04':{'status':'done','outcome':'negative numerical qualification','diagnostic_completed':True,
               'native_hardware_passed':True,'passing_recipes':[]},
        'H44':{'status':'deferred','model_comparison_executed':False,
               'reason':'Neither fixed native recipe passes repeatability. Runtime alone is insufficient; require E05 qualification and a consistent cache before fitting.'},
        'uniform_fitting_rescore_executed':False,'new_group_check_executed':False,'joint_profile_executed':False,
        'new_official_submissions':0,'best_official_score':.95481,'queued_followup':'E05',
        'serving_settings_selected_by_human_label_loss':False,'confirmation_used':False,
        'total_charged_seconds':audit['total_charged_seconds'],'within_original_caps':audit['within_original_caps'],
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256':{name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in
                       ['repeatability_results.json','diagnostic_controls.json','terminal_audit.json']}}
    for target in [folder,ROOT/'runs/span_004']:json_write(target/'decisions.json',value)
    print(json.dumps(value,indent=2))

if __name__=='__main__':main()
