extern crate stelaro;

mod test_sanity {
    #[test]
    fn test_add() {
        let result = stelaro::add(2, 2);
        assert_eq!(result, 4);
    }
}
