"""Rust/Python binding test module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: October 2024
    - License: MIT
"""

import numpy as np
import stelaro.stelaro as stelaro_rs


def test_binding_sanity():
    assert stelaro_rs.sanity() == "The Python binding functions as expected."


def test_numpy_sanity():
    x = 3
    A = np.array([1, 2, 3], dtype=np.float64)
    B = np.array([1, 1, 1], dtype=np.float64)
    expected = np.array([4, 7, 10])
    result = stelaro_rs.axb(x, A, B)
    assert (expected == result).all(), "Unexpected result"
