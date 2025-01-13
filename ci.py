"""Continuous integration suite for building and testing the project.

Usage: python3 ci.py

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: October 2024
    - License: MIT
"""

import os
import subprocess
import pathlib
import shutil


os.chdir(os.path.dirname(os.path.abspath(__file__)))

subprocess.check_call(["cargo", "fmt"])
subprocess.check_call(["cargo", "build", "--no-default-features"])
subprocess.check_call(["cargo", "build", "--features",  "opencl"])
try:
    os.remove("stelarilo")
except: pass
try:
    os.remove("stelarilo.exe")
except: pass
for executable in ("stelarilo", "stelarilo.exe"):
    try:
        shutil.copyfile(f"target/debug/{executable}", f"./{executable}")
    except: pass
subprocess.check_call(["cargo", "test", "--features",  "opencl"])
subprocess.check_call(["maturin", "develop", "--features",  "opencl"])
subprocess.check_call(["pytest", "tests"])

directories = ["stelaro", "tests"]
n_style_errors = 0
for directory in directories:
    for path in pathlib.Path(directory).rglob('*.py'):
        r = subprocess.run(
            ["pycodestyle", str(path)],
            capture_output=True, shell=True, text=True)
        n = len(r.stdout.split('\n')) - 1
        if n:
            print(r.stdout[:-1])
        n_style_errors += n
if n_style_errors > 0:
    raise RuntimeError("Style errors detected in Python files.")

print("Success: all validations passed.")
