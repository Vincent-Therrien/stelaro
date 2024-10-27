"""Continuous integration suite for building and testing the project.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: October 2024
    - License: MIT
"""

import os
import subprocess
import pathlib

os.chdir(os.path.dirname(os.path.abspath(__file__)))

subprocess.check_call(["cargo", "fmt"])
subprocess.check_call(["cargo", "build", "--features",  "opencl"])
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
