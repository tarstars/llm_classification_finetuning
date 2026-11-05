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
        self.data['active_version'] = 1
        self.data['versions'] = [
            {'number': 1, 'state': 'active', 'item_ids': [x['id'] for x in work],
             'focus_ids': ['E00', 'H09'], 'history': []},
            {'number': 2, 'state': 'draft', 'item_ids': [], 'focus_ids': [], 'history': []},
        ]
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
        hypotheses = [x for x in actual['items'] if x['kind'] == 'hypothesis']
        self.assertGreaterEqual(len(hypotheses), 40)
        self.assertTrue({f'H{i:02}' for i in range(1, 41)} <= {x['id'] for x in hypotheses})
        self.assertEqual(backlog.DEFAULT_VIEW.read_text(), backlog.render(actual))
        self.assertEqual((backlog.DEFAULT_DATA.parent / 'NEXT.md').read_text(),
                         backlog.render(actual, version=actual['active_version'] + 1, link_prefix='..'))
        backlog.check_snapshots(actual, backlog.DEFAULT_DATA)

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

    def finished_focus(self):
        data = self.finish_prerequisite()
        return backlog.update(data, 'H09', 'done', 'Pilot finished', ['docs/pilot.md'])

    def proposal(self, kind='hypothesis'):
        item = deepcopy(next(x for x in self.data['items'] if x['id'] == 'H01'))
        item.update(kind=kind, title='A follow-up grounded in the pilot',
                    gain=None if kind == 'prerequisite' else [-.01, .02], depends_on=['H09'])
        return item

    def test_discoveries_are_queued_with_provenance_and_isolated_from_active_work(self):
        original = deepcopy(self.data)
        data, key = backlog.queue(self.finished_focus(), self.proposal(), ['H09'],
                                  'The pilot suggests this test', ['docs/pilot.md'])
        self.assertEqual(key, 'H41')
        item = next(x for x in data['items'] if x['id'] == key)
        self.assertEqual((item['introduced_in'], item['discovered_during'], item['derived_from']),
                         (2, 1, ['H09']))
        self.assertNotIn(key, {x['id'] for x in backlog.select(data)})
        self.assertEqual([x['id'] for x in backlog.select(data, version=2)], [key])
        self.assertEqual(backlog.select(data, version=2, actionable=True), [])
        for status in ('ready', 'running', 'done'):
            with self.assertRaisesRegex(backlog.BacklogError, 'next-version'):
                backlog.update(data, key, status, 'Too early', ['docs/pilot.md'])
        data, second = backlog.queue(data, self.proposal(), ['H09'], 'Another independent test')
        self.assertEqual(second, 'H42')
        data, setup = backlog.queue(data, self.proposal('prerequisite'), ['H09'], 'Need setup first')
        self.assertEqual(setup, 'E01')
        self.assertEqual(self.data, original)

    def test_incomplete_proposals_and_unknown_parents_rejected_without_mutation(self):
        original = deepcopy(self.data)
        with self.assertRaises(backlog.BacklogError):
            backlog.queue(self.data, backlog.task_template(), ['H09'], 'Incomplete')
        with self.assertRaisesRegex(backlog.BacklogError, 'parents must exist'):
            backlog.queue(self.data, self.proposal(), ['H99'], 'Invalid source')
        self.assertEqual(self.data, original)

    def test_user_planning_addition_preserves_start_snapshot_and_final_history(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'backlog.json'
            data = backlog.advance(self.finished_focus(), source, 'Finished', ['docs/span.md'])
            paths = backlog.snapshot_paths(source, 2, 'start')
            original = [path.read_bytes() for path in paths]
            changed, key = backlog.queue(data, self.proposal(), ['H09'], 'User requested next-span task',
                                         ['docs/plan.md'], into_active=True)
            item = next(x for x in changed['items'] if x['id'] == key)
            self.assertEqual((item['introduced_in'], item['discovered_during'], item['status']),
                             (2, None, 'planned'))
            self.assertEqual(backlog.version_record(changed, 3)['item_ids'], [])
            self.assertEqual(backlog.version_record(changed)['focus_ids'], [])
            backlog.check_snapshots(changed, source)
            self.assertEqual([path.read_bytes() for path in paths], original)
            self.assertNotIn(key, backlog.version_record(data)['item_ids'])
            changed = backlog.set_focus(changed, 2, [key], 'Select the user-requested pilot')
            changed = backlog.update(changed, key, 'done', 'Actual result', ['docs/result.md'])
            changed = backlog.advance(changed, source, 'Finished pilot', ['docs/span2.md'])
            backlog.check_snapshots(changed, source)
            self.assertIn(key, backlog.version_record(backlog.load_version(changed, source, 2), 2)['item_ids'])
            self.assertEqual([path.read_bytes() for path in paths], original)

    def test_active_additions_reject_selected_or_started_spans_and_unlogged_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'backlog.json'
            data = backlog.advance(self.finished_focus(), source, 'Finished', ['docs/span.md'])
            original = deepcopy(data)
            with self.assertRaisesRegex(backlog.BacklogError, '--evidence'):
                backlog.queue(data, self.proposal(), ['H09'], 'No evidence', into_active=True)
            focused = backlog.set_focus(data, 2, ['H04'], 'Selected')
            with self.assertRaisesRegex(backlog.BacklogError, 'unstarted'):
                backlog.queue(focused, self.proposal(), ['H09'], 'Too late', ['docs/plan.md'], into_active=True)
            worked = backlog.update(data, 'H04', 'running', 'Started preparation')
            worked = backlog.update(worked, 'H04', 'deferred', 'Stopped', ['docs/stopped.md'])
            with self.assertRaisesRegex(backlog.BacklogError, 'unstarted'):
                backlog.queue(worked, self.proposal(), ['H09'], 'Stopped is not unstarted',
                              ['docs/plan.md'], into_active=True)
            changed, _ = backlog.queue(data, self.proposal(), ['H09'], 'User planning',
                                       ['docs/plan.md'], into_active=True)
            backlog.version_record(changed)['history'] = []
            with self.assertRaisesRegex(backlog.BacklogError, 'membership changed'):
                backlog.check_snapshots(changed, source)
            self.assertEqual(data, original)

    def test_cli_active_planning_addition_is_visible_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'backlog.json'
            data = backlog.advance(self.finished_focus(), source, 'Finished', ['docs/span.md'])
            backlog.save(data, source, source.with_suffix('.md'))
            proposal = Path(directory) / 'proposal.json'
            proposal.write_text(json.dumps(self.proposal()))
            def run(*args):
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    backlog.main(['--file', str(source), *args])
                return out.getvalue()
            self.assertIn('Added H41 to unstarted v2', run(
                'queue', '--into-active', '--item-file', str(proposal), '--from', 'H09',
                '--note', 'User requested upcoming-span planning', '--evidence', 'docs/plan.md'))
            self.assertIn('H41', {x['id'] for x in json.loads(run('list', '--new', '--json'))})
            self.assertEqual(json.loads(run('list', '--version', 'next', '--json')), [])
            self.assertIn('OK:', run('check'))

    def test_advance_requires_resolved_focus_and_no_running_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'backlog.json'
            with self.assertRaisesRegex(backlog.BacklogError, 'not been resolved'):
                backlog.advance(self.data, source, 'Premature', ['docs/span.md'])
            running = backlog.update(self.finished_focus(), 'H04', 'running', 'Other work')
            with self.assertRaisesRegex(backlog.BacklogError, 'task is running'):
                backlog.advance(running, source, 'Orphan job', ['docs/span.md'])
            deferred = backlog.update(self.data, 'E00', 'deferred', 'Quota unavailable')
            deferred['versions'][0]['focus_ids'] = ['E00']
            with self.assertRaisesRegex(backlog.BacklogError, 'deferral needs a decision'):
                backlog.advance(deferred, source, 'No quota', ['docs/span.md'])

    def test_advance_freezes_history_carries_open_work_and_can_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'backlog.json'
            initial = deepcopy(self.data)
            initial['versions'][0].update(backlog.seal_snapshot(initial, source, 'start', 'Initial'))
            data = backlog.update(initial, 'E00', 'done', 'Protocol', ['docs/protocol.md'])
            data = backlog.update(data, 'H09', 'done', 'Pilot', ['docs/pilot.md'])
            data, key = backlog.queue(data, self.proposal(), ['H09'], 'Observed opportunity')
            next_data = backlog.advance(data, source, 'First span finished', ['docs/span.md'])
            self.assertEqual(next_data['active_version'], 2)
            self.assertEqual([x['state'] for x in next_data['versions']], ['closed', 'active', 'draft'])
            members = backlog.version_record(next_data)['item_ids']
            self.assertIn(key, members)
            self.assertIn('H07', members)  # justified later work is retained
            self.assertNotIn('H09', members)
            self.assertNotIn('E00', members)
            self.assertEqual([x['id'] for x in backlog.select(next_data, cohort='new')], [key])
            carried = {x['id'] for x in backlog.select(next_data, cohort='carried')}
            self.assertIn('H07', carried)
            self.assertNotIn(key, carried)
            self.assertEqual(next_data, backlog.advance(data, source, 'First span finished', ['docs/span.md']))
            backlog.check_snapshots(next_data, source)
            historical = backlog.load_version(next_data, source, 1)
            archived = next(x for x in historical['items'] if x['id'] == 'H07')
            self.assertEqual(archived['status'], 'deferred')
            newer = backlog.update(next_data, 'H07', 'ready', 'New resources')
            self.assertEqual(next(x for x in historical['items'] if x['id'] == 'H07')['status'], 'deferred')
            self.assertEqual(next(x for x in newer['items'] if x['id'] == 'H07')['status'], 'ready')
            with self.assertRaisesRegex(backlog.BacklogError, 'historical task is closed'):
                backlog.update(newer, 'H09', 'running', 'Rewrite the past')
            with self.assertRaisesRegex(backlog.BacklogError, 'read-only'):
                backlog.update(historical, 'H09', 'planned', 'Edit archive')
            self.assertEqual(backlog.select(historical, actionable=True), [])

    def test_snapshot_tampering_and_scope_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'backlog.json'
            data = deepcopy(self.data)
            data['versions'][0].update(backlog.seal_snapshot(data, source, 'start', 'Initial'))
            path, _ = backlog.snapshot_paths(source, 1, 'start')
            original = path.read_bytes()
            with self.assertRaisesRegex(backlog.BacklogError, 'cannot be overwritten'):
                backlog.seal_snapshot(data, source, 'start', 'Changed snapshot')
            self.assertEqual(path.read_bytes(), original)
            scope_change = deepcopy(data)
            scope_change['versions'][0]['item_ids'].remove('H40')
            with self.assertRaisesRegex(backlog.BacklogError, 'membership changed'):
                backlog.check_snapshots(scope_change, source)
            path.write_bytes(original + b'\n')
            with self.assertRaisesRegex(backlog.BacklogError, 'hash mismatch'):
                backlog.check_snapshots(data, source)

    def test_focus_is_explicit_and_draft_scope_is_not_active_scope(self):
        data = backlog.set_focus(self.data, 1, ['E00', 'H04'], 'Choose cheaper pilot')
        self.assertEqual([x['id'] for x in backlog.select(data, focus=True)], ['E00', 'H04'])
        with self.assertRaisesRegex(backlog.BacklogError, 'distinct tasks'):
            backlog.set_focus(data, 1, ['H41'], 'Not in this version')
        running = backlog.update(self.finish_prerequisite(), 'H09', 'running', 'Working')
        with self.assertRaisesRegex(backlog.BacklogError, 'running work'):
            backlog.set_focus(running, 1, ['H04'], 'Change mid-run')

    def test_cli_queue_advance_and_historical_views(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'backlog.json'
            proposal = Path(directory) / 'proposal.json'
            proposal.write_text(json.dumps(self.proposal()))
            backlog.save(self.data, source, source.with_suffix('.md'))
            def run(*args):
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    backlog.main(['--file', str(source), *args])
                return out.getvalue()
            self.assertIn('Queued H41', run('queue', '--item-file', str(proposal), '--from', 'H09',
                                           '--note', 'Pilot discovery'))
            self.assertEqual([x['id'] for x in json.loads(run('list', '--version', 'next', '--json'))], ['H41'])
            for key in ('E00', 'H09'):
                run('update', key, '--status', 'done', '--note', 'Finished', '--evidence', 'docs/report.md')
            self.assertIn('active backlog is now v2', run('advance', '--note', 'Finished span',
                                                        '--evidence', 'docs/span.md'))
            self.assertIn('v2', run('versions'))
            old_ids = {x['id'] for x in json.loads(run('list', '--version', '1', '--json'))}
            new_ids = {x['id'] for x in json.loads(run('list', '--json'))}
            self.assertIn('H09', old_ids)
            self.assertNotIn('H41', old_ids)
            self.assertIn('H41', new_ids)
            self.assertNotIn('H09', new_ids)
            self.assertIn('OK:', run('check'))

    def test_repeated_spans_keep_old_views_and_allocate_new_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'backlog.json'
            data, first = backlog.queue(self.finished_focus(), self.proposal(), ['H09'], 'First discovery')
            data = backlog.advance(data, source, 'First span', ['docs/span1.md'])
            data = backlog.set_focus(data, 2, [first], 'Follow the first discovery')
            data = backlog.update(data, first, 'done', 'Second pilot finished', ['docs/pilot2.md'])
            data, second = backlog.queue(data, self.proposal(), [first], 'Second discovery')
            self.assertEqual(second, 'H42')
            data = backlog.advance(data, source, 'Second span', ['docs/span2.md'])
            self.assertEqual(data['active_version'], 3)
            self.assertEqual([x['id'] for x in backlog.select(data, cohort='new')], [second])
            self.assertNotIn(first, {x['id'] for x in backlog.select(data)})
            self.assertIn('H02', {x['id'] for x in backlog.select(data, cohort='carried')})
            backlog.check_snapshots(data, source)
            v1 = backlog.load_version(data, source, 1)
            v2 = backlog.load_version(data, source, 2)
            self.assertNotIn(first, {x['id'] for x in backlog.select(v1)})
            self.assertIn(first, {x['id'] for x in backlog.select(v2)})
            self.assertNotIn(second, {x['id'] for x in backlog.select(v2)})


if __name__ == '__main__':
    unittest.main()
