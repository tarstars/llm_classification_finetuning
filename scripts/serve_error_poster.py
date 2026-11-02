"""Serve the generated private poster on localhost only."""
import argparse,json,os
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765)
p.add_argument('--directory',type=Path,default=Path.home()/'database/llm_classification_finetuning/eda/attempt_002/poster/site')
a=p.parse_args();root=a.directory.resolve()
assert (root/'index.html').exists()
class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control','no-store');super().end_headers()
handler=partial(Handler,directory=str(root))
server=ThreadingHTTPServer(('127.0.0.1',a.port),handler)
state={'pid':os.getpid(),'url':f'http://127.0.0.1:{server.server_port}/','directory':str(root)}
(root.parent/'server.json').write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(state),flush=True)
server.serve_forever()
