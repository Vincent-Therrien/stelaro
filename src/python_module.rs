use pyo3::prelude::*;

#[pyfunction]
fn sanity() -> PyResult<String> {
    Ok("The Python binding functions as expected.".to_string())
}

#[pymodule]
fn stelaro(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sanity, m)?)?;
    Ok(())
}
