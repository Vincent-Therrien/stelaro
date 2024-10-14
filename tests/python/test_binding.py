"""Rust/Python binding test module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: October 2024
    - License: MIT
"""

import stelaro.stelaro as stelaro_rs


def test_binding_sanity():
    assert stelaro_rs.sanity() == "The Python binding functions as expected."
