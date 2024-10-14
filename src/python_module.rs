use numpy::ndarray::{ArrayD, ArrayViewD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

// TODO: Remove this function after the binding is more developed.
#[pyfunction]
fn sanity() -> PyResult<String> {
    Ok("The Python binding functions as expected.".to_string())
}

// TODO: Remove this function after the binding is more developed.
fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
    a * &x + &y
}

// TODO: Remove this function after the binding is more developed.
#[pyfunction]
fn axb<'py>(
    py: Python<'py>,
    a: f64,
    x: PyReadonlyArrayDyn<'py, f64>,
    y: PyReadonlyArrayDyn<'py, f64>,
) -> Bound<'py, PyArrayDyn<f64>> {
    let x = x.as_array();
    let y = y.as_array();
    let z = axpy(a, x, y);
    z.into_pyarray_bound(py)
}

#[pymodule]
fn stelaro(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sanity, m)?)?;
    m.add_function(wrap_pyfunction!(axb, m)?)?;
    Ok(())
}
