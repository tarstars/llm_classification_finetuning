"""Record why a final merged public competition model is not an independent teacher."""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now


def main():
    root=ROOT/'runs/span_008/teacher_audit';names=['README.md','src__prepare_data.py','src__run_pipeline.sh','src__run_final.sh','src__run_gptq.py','src__merge_lora.py']
    texts={name:(root/name).read_text() for name in names}
    assert 'StratifiedKFold(n_splits=5' in texts['src__prepare_data.py'] and "train['fold'] != fold" in texts['src__prepare_data.py']
    assert 'lmsys-33k-deduplicated.csv' in texts['src__prepare_data.py']
    assert all('gemma2fold'+str(i) in texts['src__merge_lora.py'] for i in range(5))
    readiness=read_json(REPO/'results/span_008/source_readiness.json')
    result={'task':'H15','audit_complete':True,'teacher':'Final five-fold Gemma2-9B adapter merge from shyoulala/LMSYS_BlackPearl',
        'repository':'https://github.com/shyoulala/LMSYS_BlackPearl','commit':'05ac9b917a99aac43bfe0dbcb30e6e83a38688be','observed_at':utc_now(),
        'files_sha256':{n:sha256(root/n) for n in names},
        'protocol_observations':['Preprocessing reads original competition train.csv and adds the external deduplicated33K dataset to each training fold.',
            'Five stratified folds use seed2025, with each original training row in four fitting folds.',
            'Final source averages all five trained adapters before quantization; this is not an OOF predictor for any original training row.',
            'Published workflow distills supervised70B/72B teachers into9B, with eight A100-80GB training hardware; its quality is not evidence that an unfitted general judge is stronger.'],
        'inference':'If the published checkpoint follows this code and includes the same original competition rows as described, our development and confirmation votes have influenced four of its five component adapters. It cannot establish independent teacher advantage on these cohorts.',
        'checkpoint_training_independently_reproduced':False,'checkpoint_downloaded':False,'quality_measured':False,'distillation_performed':False,
        'decision':'Defer actual distillation until a stronger teacher is measured without supervised access to evaluation groups. Public solution is architectural evidence, not a clean benchmark or adopted checkpoint.',
        'future_pilot':'Train or obtain a teacher whose supervised training excludes our group partitions; verify hashes and provenance; compare with the student on fixed development groups; only if useful, score a fixed training-only subset and compare CE+KL against matched CE-only accepted exposure. Cost a separate confirmation and full teacher generation before launch.',
        'qwen8_availability':[v for v in readiness['qwen8_kaggle_versions'] if v.get('variationSlug')=='8b' and v.get('versionNumber')==1 and v.get('framework')=='MODEL_FRAMEWORK_TRANSFORMERS'],
        'competition_text_sent':False,'new_gpu_seconds':0,'source_sha256':sha256(Path(__file__))}
    assert len(result['qwen8_availability'])==1,'Do not infer model availability from another variation version1'
    atomic_json(REPO/'results/span_008/H15_audit.json',result);print({'teacher_audit_complete':True,'distillation_performed':False,'qwen8_exact_variation_available':True})


if __name__=='__main__':main()
