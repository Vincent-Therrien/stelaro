"""Continuous integration suite for building and testing the project.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: October 2024
    - License: MIT
"""

import subprocess

subprocess.check_call(["cargo", "build"])
subprocess.check_call(["cargo", "test"])
subprocess.check_call(["maturin", "develop"])
subprocess.check_call(["pytest", "tests"])
