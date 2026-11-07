import json
from pathlib import Path
import tempfile
import unittest
from scripts.audit_span_010_order import write_qualification


class QualificationRecords(unittest.TestCase):
    def test_attempt_names_preserve_prior_failure_and_reject_model_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            first=dict(role='order_pilot',runtime_gate_passed=False)
            second=dict(role='order_packed',runtime_gate_passed=True)
            write_qualification(directory,'order_pilot',first)
            write_qualification(directory,'order_packed',second)
            self.assertEqual(json.loads((directory/'H57_order_pilot_qualification.json').read_text()),first)
            self.assertEqual(json.loads((directory/'H57_qualification.json').read_text()),second)
            for role,result in [('m3',dict(role='m3')),('order_pilot',second)]:
                with self.assertRaises(ValueError):write_qualification(directory,role,result)
            self.assertFalse((directory/'H57_m3_qualification.json').exists())
            self.assertEqual(json.loads((directory/'H57_qualification.json').read_text()),second)


if __name__=='__main__':unittest.main()
