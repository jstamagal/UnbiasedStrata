#!/usr/bin/env python3
import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[2]
target = root / "third_party/llama.cpp"
patches = sorted((root / "third_party/llama.cpp-patches").glob("*.patch"))
if not patches:
    sys.exit("No llama.cpp patches found")
for patch in patches:
    data = patch.read_bytes()
    def run(*args):
        return subprocess.run(["patch", "-p1", "--batch", *args], cwd=target,
                              input=data, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    forward = run("--forward", "--dry-run")
    if forward.returncode == 0:
        result = run("--forward", "--no-backup-if-mismatch")
        if result.returncode:
            sys.exit(result.stdout.decode())
        print("Applied " + patch.name)
    elif run("--reverse", "--dry-run").returncode == 0:
        print("Already applied " + patch.name)
    else:
        sys.exit("Cannot apply " + patch.name + "\n" + forward.stdout.decode())
