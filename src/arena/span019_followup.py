"""Complete the fixed public comparison; hidden execution uses test text only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

from .span010_order import TEXT, digest, event, save
from .span019_pipeline import orchestrate as engineering


def orchestrate(args, launch=subprocess.run):
    if not args.qualify:
        return engineering(args, launch)
    if args.engineering_only:
        raise ValueError('The followup completes the fixed public inputs')
    source_path = Path(args.source)
    if digest(source_path) != args.source_sha256:
        raise ValueError('The later text cohort differs from the frozen input')
    selection = json.loads(Path(args.selection).read_text())
    prepared = json.loads(Path(args.routes).read_text())
    source = pd.read_parquet(source_path, columns=['id', *TEXT])
    if not source.id.is_unique or source.id.tolist() != selection['later_ids']:
        raise ValueError('Later cohort IDs or order differ from the frozen selection')
    train = pd.read_csv(Path(args.raw)/'train.csv', usecols=['id', *TEXT]).set_index('id', drop=False)
    control = train.loc[selection['control_ids']].reset_index(drop=True)
    routes = {}
    for name, frame in [('source', source), ('control', control)]:
        route = prepared['routes'][name]
        selected = frame.iloc[route['selected_indices']].reset_index(drop=True)
        if selected.id.tolist() != route['selected_ids']:
            raise ValueError('Parent route IDs differ from the frozen route')
        routes[name] = selected
    def checked_launch(command, check):
        command=list(command)
        if command[3]=='arena.span019_infer':
            command[3]='arena.span019_checked_infer'
        return launch(command, check=check)
    initial = engineering(args, checked_launch)
    out = Path(args.out)
    source_out, reverse_out = out/'source_inference', out/'parent_reverse'
    for folder in [source_out, reverse_out]:
        folder.mkdir()
    source.to_parquet(source_out/'source.parquet', index=False)
    control.iloc[:80].to_parquet(reverse_out/'warmup.parquet', index=False)
    routes['control'].to_parquet(reverse_out/'routed_control.parquet', index=False)
    # The unchanged retained worker accepts this historical cohort alias.
    routes['source'].to_parquet(reverse_out/'routed_long.parquet', index=False)
    def execute(module, arguments):
        started = time.monotonic()
        launch([sys.executable, '-u', '-m', module, *arguments], check=True)
        return time.monotonic()-started
    source_seconds = execute('arena.span019_infer', ['--mode', 'parallel', '--model', args.origin,
        '--native-source', args.native_source, '--head-sha256', args.head_sha256,
        '--out', str(source_out), '--cohorts', 'source'])
    reverse_seconds = execute('arena.span010_order', ['--role', 'm0_reverse', '--raw', args.raw,
        '--assets', args.assets, '--out', str(reverse_out),
        '--cohorts', 'warmup,routed_control,routed_long'])
    for folder in [source_out, reverse_out]:
        for path in folder.glob('*.parquet'):
            path.unlink()
    result = dict(complete=True, public_stage='fixed_quality_inputs', initial_engineering=initial,
        source_rows=len(source), source_sha256=args.source_sha256,
        source_process_seconds=source_seconds, reverse_process_seconds=reverse_seconds,
        source_routed_ids=routes['source'].id.tolist(), control_routed_ids=routes['control'].id.tolist(),
        candidate_weights=[.5, .5], temperature=1, uses_targets=False,
        quality_evaluated=False, official_submission_eligible=False)
    save(out/'followup.json', result)
    event('mirror_followup_complete', complete=True, rows=len(source))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['raw', 'assets', 'origin', 'native-source', 'head-sha256', 'out']:
        parser.add_argument('--'+name, required=True)
    for name in ['selection', 'source', 'source-sha256', 'routes']:
        parser.add_argument('--'+name)
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--engineering-only', action='store_true')
    orchestrate(parser.parse_args())
