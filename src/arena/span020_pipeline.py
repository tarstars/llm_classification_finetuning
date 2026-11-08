"""The fixed H66batch recipe, reusing target-free public/hidden orchestration."""
import argparse
import subprocess

from .span019_followup import orchestrate as complete_inputs


def orchestrate(args,launch=subprocess.run):
    def candidate_launch(command,check):
        command=list(command)
        position=command.index('-m')+1
        if command[position] in ['arena.span019_infer','arena.span019_checked_infer']:
            command[position]='arena.span020_infer'
        return launch(command,check=check)
    return complete_inputs(args,candidate_launch)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['raw','assets','origin','native-source','head-sha256','out']:
        parser.add_argument('--'+name,required=True)
    for name in ['selection','source','source-sha256','routes']:
        parser.add_argument('--'+name)
    parser.add_argument('--qualify',action='store_true')
    parser.add_argument('--engineering-only',action='store_true')
    orchestrate(parser.parse_args())
