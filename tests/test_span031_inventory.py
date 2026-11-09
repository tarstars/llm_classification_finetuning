"""Exercise coverage and ownership boundaries of the actual draft resolver."""
from pathlib import Path
from types import SimpleNamespace as Row
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.span031_inventory import resolve_profile


class ResolutionTest(unittest.TestCase):
    def setUp(self):
        self.ref='arseniytrushin/rejected'
        self.stub=Row(ref='',id=0,author='',slug='',current_version_number=0,
            title='[Private Notebook]',last_run_time='2010-04-01T00:00:00Z')
        self.profile=[self.stub,Row(ref='arseniytrushin/known')]
        self.owned=[Row(ref=self.ref,owner='arseniytrushin'),Row(ref='arseniytrushin/known',owner='arseniytrushin')]

    def test_recovers_every_entry_without_dropping_placeholder(self):
        rows,proof=resolve_profile(self.profile,self.owned,self.ref)
        self.assertEqual([x.ref for x in rows],[self.ref,'arseniytrushin/known'])
        self.assertFalse(proof['placeholder_dropped'])

    def test_missing_or_replayed_owned_catalog_rejected(self):
        for bad in [self.owned[:1],[self.owned[0],self.owned[0]]]:
            with self.assertRaises(AssertionError):resolve_profile(self.profile,bad,self.ref)

    def test_wrong_owner_or_different_missing_identity_rejected(self):
        for bad in [[Row(ref=self.ref,owner='another'),self.owned[1]],
                    [Row(ref='arseniytrushin/other',owner='arseniytrushin'),self.owned[1]]]:
            with self.assertRaises(AssertionError):resolve_profile(self.profile,bad,self.ref)

    def test_unknown_redaction_or_recent_stub_rejected(self):
        for name,value in [('title','Unknown'),('last_run_time','2026-11-09'),('id',123)]:
            original=getattr(self.stub,name);setattr(self.stub,name,value)
            with self.assertRaises(AssertionError):resolve_profile(self.profile,self.owned,self.ref)
            setattr(self.stub,name,original)


if __name__=='__main__':unittest.main()
