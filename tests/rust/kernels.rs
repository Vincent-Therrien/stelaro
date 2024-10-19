#[cfg(test)]
mod test_kernels {
    #[cfg(feature = "opencl")]
    #[test]
    fn test_kernel() {
        let _a = stelaro::kernels::test::trivial();
    }
}
