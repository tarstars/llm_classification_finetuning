import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts'))
from submit_span_011_calibration import candidate
from arena.experiment import sha256


class OfficialAdmission(unittest.TestCase):
    def test_blocks_unconfirmed_stale_or_unreviewed_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pub=root/'public';pub.mkdir();role='calibration_confirmation'
            notebook=root/'notebooks'/role/'job.ipynb';notebook.parent.mkdir(parents=True);notebook.write_text('frozen-notebook')
            job=root/'jobs'/role;(job/'output').mkdir(parents=True)
            csv=job/'output/submission.csv';csv.write_text('id,a,b,t\n1,0.7,0.2,0.1\n')
            evidence={
                'confirmation_metrics.json':dict(information_gate_passed=True,temperature_fit_on_confirmation=False,
                    temperature=1.1,comparison=dict(loss_reduction=.002,paired_group95=[.001,.003])),
                'frozen_temperature.json':dict(learning_gate_passed=True,temperature=1.1),
                'learning_metrics.json':dict(information_gate_passed=True),
                'calibration_learning_qualification.json':dict(all_controls_passed=True,runtime_gate_passed=True),
                'calibration_confirmation_qualification.json':dict(complete=True,all_controls_passed=True,runtime_gate_passed=True,
                    csv_verified=True,both_t4_verified=True,csv_sha256=sha256(csv),runtime=dict(projection_seconds=30000),
                    source=dict(kernel_version='owner/frozen/1',notebook_sha256=sha256(notebook))),
                'calibration_confirmation_bundle.json':dict(temperature=1.1,source_sha256={},notebook_sha256=sha256(notebook)),
                'calibration_confirmation_source_verification.json':dict(passed=True,kernel_version='owner/frozen/1',notebook_sha256=sha256(notebook))}
            def write(name,data):
                path=pub/name;path.write_text(json.dumps(data));return sha256(path)
            for name,data in evidence.items():write(name,data)
            bundle=evidence['calibration_confirmation_bundle.json']
            bundle['frozen_temperature_sha256']=sha256(pub/'frozen_temperature.json');write('calibration_confirmation_bundle.json',bundle)
            decision=dict(submit=True,domain_review_completed=True,parameters_selected_without_leaderboard_feedback=True,temperature=1.1)
            def freeze_decision():
                decision['evidence_sha256']={name:sha256(pub/name) for name in evidence};write('deployment_decision.json',decision)
            freeze_decision();(job/'push.json').write_text(json.dumps(dict(ref='/code/owner/frozen',version_number=1)))
            with patch('submit_span_011_calibration.verify_sources'):
                self.assertEqual(candidate(root,pub)['temperature'],1.1)
                decision['domain_review_completed']=False;freeze_decision()
                with self.assertRaises(AssertionError):candidate(root,pub)
                decision['domain_review_completed']=True;freeze_decision()
                metrics=evidence['confirmation_metrics.json'];metrics['comparison']['loss_reduction']=.0009
                write('confirmation_metrics.json',metrics);freeze_decision()
                with self.assertRaises(AssertionError):candidate(root,pub)
                metrics['comparison']['loss_reduction']=.002;metrics['comparison']['paired_group95']=[-.001,.004]
                write('confirmation_metrics.json',metrics);freeze_decision()
                with self.assertRaises(AssertionError):candidate(root,pub)
                metrics['comparison']['paired_group95']=[.001,.003];write('confirmation_metrics.json',metrics);freeze_decision()
                metrics['temperature']=1.2;write('confirmation_metrics.json',metrics)
                with self.assertRaises(AssertionError):candidate(root,pub)  # Changed evidence cannot reuse admission.
                metrics['temperature']=1.1;write('confirmation_metrics.json',metrics);freeze_decision()
                csv.write_text('modified output')
                with self.assertRaises(AssertionError):candidate(root,pub)


if __name__=='__main__':unittest.main()
