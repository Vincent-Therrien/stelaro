use pyo3::prelude::*;
use crate::add_ab;

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok(add_ab(a, b).to_string())
}

#[pymodule]
fn stelaro_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
