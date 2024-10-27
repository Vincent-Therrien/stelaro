#[cfg(feature = "opencl")]
extern crate ocl;

#[macro_use]
extern crate lazy_static;

mod python_module;
pub mod io {
    pub mod format;
    pub mod sequence;
}

#[cfg(feature = "opencl")]
pub mod kernels {
    pub mod test; // TODO: Use real modules.
}
