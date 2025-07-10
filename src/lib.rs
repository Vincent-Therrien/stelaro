#[cfg(feature = "opencl")]
extern crate ocl;

#[macro_use]
extern crate lazy_static;

pub mod io;

pub mod data;

pub mod transform;

pub mod utils {
    pub mod progress;
}

#[cfg(feature = "python")]
mod python_module;

#[cfg(feature = "opencl")]
pub mod kernels {
    pub mod test; // TODO: Use real modules.
}
