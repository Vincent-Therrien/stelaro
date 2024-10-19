#[cfg(test)]
mod test_kernels {
    #[cfg(feature = "opencl")]
    #[test]
    fn test_kernel() {
        let a = stelaro::trivial();
    }
}
