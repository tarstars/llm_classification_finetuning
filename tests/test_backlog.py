"""Research workflow checks: dependency gates, safe updates and generated memory."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arena import backlog


class BacklogWorkflow(unittest.TestCase):
    def setUp(self):
        self.data = backlog.read()
        # A deterministic workflow fixture: real backlog status/estimate edits must
        # not break unit tests. The actual source/view consistency is checked below.
        template = deepcopy(next(x for x in self.data['items'] if x['kind'] == 'hypothesis'))
        work = []
        for index, key in enumerate(['E00'] + [f'H{i:02}' for i in range(1, 41)]):
            item = deepcopy(template)
            item.update(id=key, kind='prerequisite' if key == 'E00' else 'hypothesis',
                        rank=index + 1, priority='P0' if key == 'E00' else 'P2',
                        status='ready' if key == 'E00' else 'planned',
                        gain=None if key == 'E00' else [-.01, .01],
                        risk='medium', engineering_hours=[1, 2], history=[],
                        depends_on=[] if key == 'E00' else ['E00'])
            work.append(item)
        self.data['items'] = work
        by_id = {x['id']: x for x in work}
        by_id['H07']['status'] = 'deferred'
        by_id['H15']['depends_on'] = ['E00', 'H01']
        by_id['H15']['gain'] = [-.02, .05]
        for key, eng, gain, risk in [('H01', [6,16], [-.015,.04], 'high'),
                                     ('H04', [4,10], [-.01,.025], 'medium'),
                                     ('H09', [6,14], [-.01,.02], 'medium'),
                                     ('H11', [6,14], [-.01,.015], 'medium')]:
            by_id[key].update(priority='P1', engineering_hours=eng, gain=gain, risk=risk)

    def finish_prerequisite(self):
        return backlog.update(self.data, "E00", "done", "Protocol frozen", ["docs/protocol.md"])

    def test_forty_stable_ids_and_current_generated_page(self):
        actual = backlog.read()
        hypotheses = backlog.select(actual, kind="hypothesis")
        self.assertEqual(len(hypotheses), 40)
        self.assertEqual({x['id'] for x in hypotheses}, {f'H{i:02}' for i in range(1, 41)})
        self.assertEqual(backlog.DEFAULT_VIEW.read_text(), backlog.render(actual))

    def test_dependency_gates_ready_and_running_without_mutating_input(self):
        original = deepcopy(self.data)
        for status in ("ready", "running"):
            with self.assertRaisesRegex(backlog.BacklogError, "unfinished dependencies"):
                backlog.update(self.data, "H09", status, "Start pilot")
        self.assertEqual(self.data, original)

    def test_closed_dependency_enables_pilot_and_records_history(self):
        changed = backlog.update(self.finish_prerequisite(), "H09", "running", "Paired pilot started")
        item = next(x for x in changed['items'] if x['id'] == 'H09')
        self.assertEqual(item['status'], 'running')
        self.assertEqual(item['history'][-1]['note'], 'Paired pilot started')
        self.assertIn('+00:00', item['history'][-1]['at'])
        self.assertEqual(next(x for x in self.data['items'] if x['id'] == 'H09')['status'], 'planned')

    def test_cannot_reopen_prerequisite_of_running_work(self):
        changed = backlog.update(self.finish_prerequisite(), 'H09', 'running', 'Start')
        with self.assertRaisesRegex(backlog.BacklogError, "unfinished dependencies"):
            backlog.update(changed, 'E00', 'planned', 'Reopen protocol')

    def test_closure_requires_note_and_result_reference(self):
        for status in ('done', 'rejected'):
            with self.assertRaisesRegex(backlog.BacklogError, '--evidence'):
                backlog.update(self.data, 'E00', status, 'Finished')
        with self.assertRaisesRegex(backlog.BacklogError, 'note'):
            backlog.update(self.data, 'E00', 'done', ' ', ['docs/protocol.md'])
        with self.assertRaisesRegex(backlog.BacklogError, 'Unknown item'):
            backlog.update(self.data, 'H99', 'planned', 'Typo')
        with self.assertRaisesRegex(backlog.BacklogError, 'unfinished dependencies'):
            backlog.update(self.data, 'H09', 'done', 'Cannot bypass gate', ['results/pilot.json'])

    def test_actionable_filter_excludes_unmet_and_deferred_work(self):
        self.assertEqual([x['id'] for x in backlog.select(self.data, actionable=True)], ['E00'])
        actionable = {x['id'] for x in backlog.select(self.finish_prerequisite(), actionable=True)}
        self.assertIn('H09', actionable)
        self.assertNotIn('H15', actionable)
        self.assertNotIn('H07', actionable)
        self.assertNotIn('E00', actionable)

    def test_filters_and_sorts_have_documented_meaning(self):
        selected = backlog.select(self.data, priorities=['P1'], risks=['medium'],
                                  max_engineering=14, min_gain=.015, sort='effort')
        self.assertEqual([x['id'] for x in selected], ['H04', 'H09', 'H11'])
        optimistic = backlog.select(self.data, kind='hypothesis', sort='gain')
        self.assertEqual(optimistic[0]['id'], 'H15')
        self.assertGreaterEqual(optimistic[0]['gain'][1], optimistic[-1]['gain'][1])

    def test_cycles_unknown_dependencies_and_duplicate_ids_rejected(self):
        cyclic = deepcopy(self.data)
        cyclic['items'][0]['status'] = 'planned'
        cyclic['items'][0]['depends_on'] = ['H09']
        with self.assertRaisesRegex(backlog.BacklogError, 'cycle'):
            backlog.validate(cyclic)
        unknown = deepcopy(self.data)
        unknown['items'][1]['depends_on'] = ['H99']
        with self.assertRaisesRegex(backlog.BacklogError, 'unknown/self'):
            backlog.validate(unknown)
        duplicate = deepcopy(self.data)
        duplicate['items'].append(deepcopy(duplicate['items'][1]))
        with self.assertRaisesRegex(backlog.BacklogError, 'Duplicate item'):
            backlog.validate(duplicate)

    def test_missing_hypothesis_reversed_or_nonfinite_estimates_rejected(self):
        missing = deepcopy(self.data)
        missing['items'].pop()
        with self.assertRaisesRegex(backlog.BacklogError, 'forty'):
            backlog.validate(missing)
        for field, numbers in [('gain', [.01, -.01]), ('gain', [0, float('nan')]),
                               ('pilot_gpu_wall_hours', [-1, 2]),
                               ('engineering_hours', [True, 2])]:
            invalid = deepcopy(self.data)
            invalid['items'][1][field] = numbers
            with self.subTest(field=field, numbers=numbers), self.assertRaises(backlog.BacklogError):
                backlog.validate(invalid)
        for malformed in ([], {'items': []}, {**self.data, 'items': [{'id': []}]}):
            with self.subTest(malformed=type(malformed)), self.assertRaises(backlog.BacklogError):
                backlog.validate(malformed)

    def test_cli_lifecycle_and_stale_view_detection_in_temporary_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'backlog.json'
            view = source.with_suffix('.md')
            source.write_text(json.dumps(self.data))
            def run(*args):
                with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
                    backlog.main(['--file', str(source), *args])
                return out.getvalue()
            run('render')
            self.assertIn('OK:', run('check'))
            result = json.loads(run('list', '--actionable', '--json'))
            self.assertEqual([x['id'] for x in result], ['E00'])
            before = source.read_bytes()
            with self.assertRaises(SystemExit) as invalid:
                run('update', 'H09', '--status', 'running', '--note', 'Too early')
            self.assertEqual(invalid.exception.code, 2)
            self.assertEqual(source.read_bytes(), before)
            run('update', 'E00', '--status', 'done', '--note', 'Protocol frozen',
                '--evidence', 'docs/protocol.md')
            self.assertIn('OK:', run('check'))
            self.assertEqual(json.loads(run('show', 'E00', '--json'))['status'], 'done')
            view.write_text('stale')
            with self.assertRaises(SystemExit) as stale:
                run('check')
            self.assertEqual(stale.exception.code, 2)
            run('render')
            self.assertEqual(view.read_text(), backlog.render(backlog.read(source)))


if __name__ == '__main__':
    unittest.main()
