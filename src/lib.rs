#[cfg(feature = "opencl")]
extern crate ocl;

mod python_module;
pub mod io {
    pub mod sequence;
}

#[cfg(feature = "opencl")]
pub mod kernels {
    pub mod test; // TODO: Use real modules.
}
