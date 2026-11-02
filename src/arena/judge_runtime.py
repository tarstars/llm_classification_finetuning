"""Compile a pinned llama.cpp runtime and probe official Gemma GGUF models on Kaggle."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

import requests


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def runtime_environment(executable):
    environment = os.environ.copy()
    paths = [str(Path(executable).parent), '/usr/local/cuda/lib64', '/usr/local/cuda/targets/x86_64-linux/lib']
    environment['LD_LIBRARY_PATH'] = ':'.join(paths) + ':' + environment.get('LD_LIBRARY_PATH', '')
    return environment


def install_runtime(distribution, out):
    """Install an official release asset only after checking its published digest."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path('/tmp/arena_llama_binary')
    root.mkdir(exist_ok=True)
    archive = root / 'runtime.tar.gz'
    print('RUNTIME_BINARY', json.dumps(distribution), flush=True)
    with urllib.request.urlopen(distribution['url'], timeout=120) as response:
        with archive.open('wb') as stream: shutil.copyfileobj(response, stream)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == distribution['sha256'], 'Release digest mismatch'
    with tarfile.open(archive) as tar: tar.extractall(root, filter='data')
    executable = next(root.rglob('llama-server'))
    binary = out / 'bin'
    binary.mkdir(exist_ok=True)
    for path in executable.parent.iterdir():
        if path.name == 'llama-server' or '.so' in path.name:
            shutil.copy2(path, binary / path.name, follow_symlinks=True)
    executable = binary / 'llama-server'
    executable.chmod(0o755)
    info = {**distribution, 'version': subprocess.check_output([str(executable), '--version'],
        stderr=subprocess.STDOUT, text=True, env=runtime_environment(executable)).strip()}
    write_json(out / 'runtime.json', info)
    return executable


def build_runtime(revision, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    print('RUNTIME_SETUP', json.dumps({'revision': revision}), flush=True)
    root = Path('/tmp/arena_llama_build')
    root.mkdir(exist_ok=True)
    archive = root / 'source.tar.gz'
    with urllib.request.urlopen(f'https://github.com/ggml-org/llama.cpp/archive/{revision}.tar.gz', timeout=120) as response:
        with archive.open('wb') as stream:
            shutil.copyfileobj(response, stream)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter='data')
    source = root / ('llama.cpp-' + revision)
    build = root / 'build'
    # Kaggle exposes libcuda.so.1 without the unversioned developer symlink that
    # FindCUDAToolkit expects. Supply the actual installed driver library.
    libraries = subprocess.check_output(['ldconfig', '-p'], text=True)
    drivers = re.findall(r'libcuda\.so(?:\.1)?\s+\([^\n]+\)\s+=>\s+(\S+)', libraries)
    drivers += [str(p) for directory in ['/usr/lib64-nvidia', '/usr/lib/x86_64-linux-gnu', '/usr/local/nvidia/lib64']
                for p in Path(directory).glob('libcuda.so*')]
    driver = next((p for p in drivers if Path(p).exists()), None)
    assert driver, 'The allocated GPU driver library is not visible'
    print('CUDA_DRIVER', driver, flush=True)
    commands = [
        ['cmake', '-S', str(source), '-B', str(build), '-DGGML_CUDA=ON',
         '-DCMAKE_CUDA_ARCHITECTURES=75', '-DCMAKE_BUILD_TYPE=Release',
         '-DCUDA_cuda_driver_LIBRARY=' + driver,
         '-DLLAMA_CURL=OFF', '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF',
         '-DLLAMA_BUILD_SERVER=ON', '-DBUILD_SHARED_LIBS=ON'],
        ['cmake', '--build', str(build), '--config', 'Release', '-j', '4', '--target', 'llama-server'],
    ]
    started = time.monotonic()
    with (out / 'build.log').open('w') as log:
        for command in commands:
            print('BUILD', command, flush=True)
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                print((out / 'build.log').read_text()[-12000:], flush=True)
                raise RuntimeError('Pinned llama.cpp build failed')
    binary = out / 'bin'
    binary.mkdir(exist_ok=True)
    for path in (build / 'bin').iterdir():
        if path.name == 'llama-server' or '.so' in path.name:
            shutil.copy2(path, binary / path.name, follow_symlinks=True)
    executable = binary / 'llama-server'
    assert executable.exists()
    info = {'revision': revision, 'source_sha256': digest, 'build_seconds': time.monotonic() - started,
            'cuda_architectures': [75], 'cuda_driver_library': driver, 'version': subprocess.check_output([str(executable), '--version'],
                stderr=subprocess.STDOUT, text=True).strip()}
    write_json(out / 'runtime.json', info)
    return executable


def model_file(source):
    candidates = [Path('/kaggle/input/models') / source, Path('/kaggle/input') / '/'.join(source.split('/')[1:])]
    for root in candidates:
        if root.exists():
            files = [p for p in root.rglob('*.gguf') if p.stat().st_size > 1_000_000_000 and 'mmproj' not in p.name.lower()]
            if files:
                return max(files, key=lambda p: p.stat().st_size)
    raise FileNotFoundError('Attached GGUF model not found: ' + source)


def start_server(executable, model, out, *, slots=4, context=2048, port=8080,
                 batch_size=2048, ubatch_size=256, http_threads=32):
    out = Path(out)
    log = (out / 'server.log').open('w')
    command = [str(executable), '-m', str(model), '--host', '127.0.0.1', '--port', str(port),
               '-ngl', 'all', '--split-mode', 'layer', '--tensor-split', '1,1',
               '--ctx-size', str(slots * context), '--parallel', str(slots),
               '--batch-size', str(batch_size), '--ubatch-size', str(ubatch_size), '--flash-attn', 'on',
               '--threads-http', str(http_threads),
               '--threads', '4', '--threads-batch', '4', '--jinja']
    environment = runtime_environment(executable)
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    url = f'http://127.0.0.1:{port}'
    started = time.monotonic()
    while time.monotonic() - started < 900:
        if process.poll() is not None:
            log.close()
            raise RuntimeError('llama-server failed: ' + (out / 'server.log').read_text()[-12000:])
        try:
            if requests.get(url + '/health', timeout=2).ok:
                return process, log, url
        except requests.RequestException:
            pass
        time.sleep(2)
    process.terminate()
    log.close()
    raise TimeoutError('Model loading exceeded 15 minutes')


def post(url, path, payload):
    response = requests.post(url + path, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def smoke_probe(executable, source, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = model_file(source)
    process, log, url = start_server(executable, model, out)
    try:
        labels = {text: post(url, '/tokenize', {'content': text, 'add_special': False}) for text in ['A', 'B', 'C']}
        messages = [{'role': 'system', 'content': 'Judge two answers. Output A if answer A is better, B if answer B is better, C if tied. Output only one letter.'},
                    {'role': 'user', 'content': 'Question: What is 2+2?\nAnswer A: 4\nAnswer B: 5'}]
        template = post(url, '/apply-template', {'messages': messages, 'add_generation_prompt': True,
            'chat_template_kwargs': {'enable_thinking': False}})
        prompt = template['prompt']
        thought = '<|channel>thought\n<channel|>'
        if not prompt.endswith(thought):
            prompt += thought
        result = post(url, '/completion', {'prompt': prompt, 'n_predict': 1, 'temperature': 1.0,
            'samplers': ['temperature'], 'grammar': 'root ::= "A" | "B" | "C"', 'n_probs': 3,
            'post_sampling_probs': True, 'return_tokens': True, 'seed': 20261102})
        summary = {'model_source': source, 'model_filename': model.name, 'model_bytes': model.stat().st_size,
                   'label_tokens': labels, 'template': template, 'response': result,
                   'gpu': subprocess.check_output(['nvidia-smi'], text=True)}
        write_json(out / 'smoke.json', summary)
        print('JUDGE_SMOKE', json.dumps(summary), flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
    return summary
