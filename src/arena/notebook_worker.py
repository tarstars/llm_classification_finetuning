"""Testable notebook subprocess wrapper for future builders; historical cells stay frozen."""


def worker_cell(role,setup_source):
    """Setup defines command/env; notebook supplies WORK, STARTED and DEADLINE."""
    if not isinstance(role,str) or not role:raise ValueError('A literal role is required')
    return f'''import os,sys,time,json,signal,threading,subprocess
{setup_source}
process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env,start_new_session=True)
def stop_owned_group():
    if process.poll() is None:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
timer=threading.Timer(max(.01,DEADLINE-time.time()),stop_owned_group);timer.daemon=True;timer.start()
try:
    with (WORK/'worker_console.log').open('w') as log:
        for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
    status=process.wait(timeout=max(.01,DEADLINE-time.time()))
finally:
    timer.cancel();stop_owned_group()
    if process.stdout is not None:process.stdout.close()
record={{'seconds':time.time()-STARTED,'exit_code':status,'role':{role!r}}}
temporary=WORK/'session_summary.json.tmp';temporary.write_text(json.dumps(record,indent=2,allow_nan=False));temporary.replace(WORK/'session_summary.json')
assert status==0,'Preserve failed execution; never automatically repeat a mutation'
'''
