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
import time

start = time.time()
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("Formatting the Rust source code.")
subprocess.check_call(["cargo", "fmt"])
print("Minimal build of the Rust source code.")
subprocess.check_call(["cargo", "build", "--no-default-features"])
print("Complete build of the Rust source code.")
subprocess.check_call(["cargo", "build", "--features",  "opencl"])
print("Placing the executable program at a more easily accessible location.")
try:
    os.remove("stelarilo")
except:
    pass
try:
    os.remove("stelarilo.exe")
except:
    pass
for executable in ("stelarilo", "stelarilo.exe"):
    try:
        shutil.copyfile(f"target/debug/{executable}", f"./{executable}")
    except: pass
print("Running Rust tests.")
subprocess.check_call(["cargo", "test", "--features",  "opencl"])
print("Building the Python binding.")
subprocess.check_call(["maturin", "develop", "--features",  "opencl"])
print("Testing the Python binding.")
subprocess.check_call(["pytest", "tests"])

print("Formatting the Python source code.")
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

end = time.time()
duration = end - start
print(f"Success: all validations passed. Duration: {duration:.3} s.")
