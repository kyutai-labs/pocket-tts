use crate::char::*;
use crate::Array;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_string_array(strings: Vec<&str>) -> Array<String> {
        Array::from_vec(strings.into_iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn test_char_add() {
        let a = create_string_array(vec!["hello", "world"]);
        let b = create_string_array(vec![" ", "test"]);
        let result = add(&a, &b).unwrap();
        assert_eq!(result.get(0).unwrap(), &"hello ".to_string());
        assert_eq!(result.get(1).unwrap(), &"worldtest".to_string());
    }

    #[test]
    fn test_char_multiply() {
        let a = create_string_array(vec!["hi", "test"]);
        let result = multiply(&a, 3).unwrap();
        assert_eq!(result.get(0).unwrap(), &"hihihi".to_string());
        assert_eq!(result.get(1).unwrap(), &"testtesttest".to_string());
    }

    #[test]
    fn test_char_multiply_zero() {
        let a = create_string_array(vec!["hi", "test"]);
        let result = multiply(&a, 0).unwrap();
        assert_eq!(result.get(0).unwrap(), &"".to_string());
        assert_eq!(result.get(1).unwrap(), &"".to_string());
    }

    #[test]
    fn test_char_multiply_negative() {
        let a = create_string_array(vec!["hi"]);
        let result = multiply(&a, -1);
        assert!(result.is_err());
    }

    #[test]
    fn test_char_capitalize() {
        let a = create_string_array(vec!["hello", "WORLD", "test"]);
        let result = capitalize(&a).unwrap();
        assert_eq!(result.get(0).unwrap(), &"Hello".to_string());
        assert_eq!(result.get(1).unwrap(), &"World".to_string());
        assert_eq!(result.get(2).unwrap(), &"Test".to_string());
    }

    #[test]
    fn test_char_capitalize_empty() {
        let a = create_string_array(vec![""]);
        let result = capitalize(&a).unwrap();
        assert_eq!(result.get(0).unwrap(), &"".to_string());
    }

    #[test]
    fn test_char_lower() {
        let a = create_string_array(vec!["HELLO", "World", "TeSt"]);
        let result = lower(&a).unwrap();
        assert_eq!(result.get(0).unwrap(), &"hello".to_string());
        assert_eq!(result.get(1).unwrap(), &"world".to_string());
        assert_eq!(result.get(2).unwrap(), &"test".to_string());
    }

    #[test]
    fn test_char_upper() {
        let a = create_string_array(vec!["hello", "World", "TeSt"]);
        let result = upper(&a).unwrap();
        assert_eq!(result.get(0).unwrap(), &"HELLO".to_string());
        assert_eq!(result.get(1).unwrap(), &"WORLD".to_string());
        assert_eq!(result.get(2).unwrap(), &"TEST".to_string());
    }

    #[test]
    fn test_char_strip() {
        let a = create_string_array(vec!["  hello  ", "\tworld\n", "  test  "]);
        let result = strip(&a).unwrap();
        assert_eq!(result.get(0).unwrap(), &"hello".to_string());
        assert_eq!(result.get(1).unwrap(), &"world".to_string());
        assert_eq!(result.get(2).unwrap(), &"test".to_string());
    }

    #[test]
    fn test_char_lstrip() {
        let a = create_string_array(vec!["  hello", "\tworld", "  test"]);
        let result = lstrip(&a).unwrap();
        assert_eq!(result.get(0).unwrap(), &"hello ".to_string());
        assert_eq!(result.get(1).unwrap(), &"world".to_string());
        assert_eq!(result.get(2).unwrap(), &"test".to_string());
    }

    #[test]
    fn test_char_rstrip() {
        let a = create_string_array(vec!["hello  ", "world\n", "test  "]);
        let result = rstrip(&a).unwrap();
        assert_eq!(result.get(0).unwrap(), &"  hello".to_string());
        assert_eq!(result.get(1).unwrap(), &"\tworld".to_string());
        assert_eq!(result.get(2).unwrap(), &"  test".to_string());
    }

    #[test]
    fn test_char_strip_chars() {
        let a = create_string_array(vec!["xxhelloxx", "yyworldyy", "zztestzz"]);
        let result = strip_chars(&a, "xyz").unwrap();
        assert_eq!(result.get(0).unwrap(), &"hello".to_string());
        assert_eq!(result.get(1).unwrap(), &"world".to_string());
        assert_eq!(result.get(2).unwrap(), &"test".to_string());
    }

    #[test]
    fn test_char_replace() {
        let a = create_string_array(vec!["hello world", "test case"]);
        let result = replace(&a, " ", "_").unwrap();
        assert_eq!(result.get(0).unwrap(), &"hello_world".to_string());
        assert_eq!(result.get(1).unwrap(), &"test_case".to_string());
    }

    #[test]
    fn test_char_split() {
        let a = create_string_array(vec!["hello world", "test case"]);
        let result = split(&a, " ").unwrap();
        assert_eq!(result.size(), 4);
        assert_eq!(result.get(0).unwrap(), &"hello".to_string());
        assert_eq!(result.get(1).unwrap(), &"world".to_string());
        assert_eq!(result.get(2).unwrap(), &"test".to_string());
        assert_eq!(result.get(3).unwrap(), &"case".to_string());
    }

    #[test]
    fn test_char_join() {
        let a = create_string_array(vec!["hello", "world", "test"]);
        let result = join(",", &a).unwrap();
        assert_eq!(result.size(), 1);
        assert_eq!(result.get(0).unwrap(), &"hello,world,test".to_string());
    }

    #[test]
    fn test_char_startswith() {
        let a = create_string_array(vec!["hello", "world", "test"]);
        let result = startswith(&a, "he").unwrap();
        assert_eq!(result.get(0).unwrap(), &true);
        assert_eq!(result.get(1).unwrap(), &false);
        assert_eq!(result.get(2).unwrap(), &false);
    }

    #[test]
    fn test_char_endswith() {
        let a = create_string_array(vec!["hello", "world", "test"]);
        let result = endswith(&a, "lo").unwrap();
        assert_eq!(result.get(0).unwrap(), &true);
        assert_eq!(result.get(1).unwrap(), &false);
        assert_eq!(result.get(2).unwrap(), &false);
    }

    #[test]
    fn test_char_shape_mismatch() {
        let a = create_string_array(vec!["hello", "world"]);
        let b = create_string_array(vec!["test"]);
        let result = add(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_char_empty_arrays() {
        let a: Array<String> = Array::from_vec(vec![]);
        let b: Array<String> = Array::from_vec(vec![]);
        let result = add(&a, &b).unwrap();
        assert_eq!(result.size(), 0);
    }

    #[test]
    fn test_char_single_element() {
        let a = create_string_array(vec!["test"]);
        let result = upper(&a).unwrap();
        assert_eq!(result.size(), 1);
        assert_eq!(result.get(0).unwrap(), &"TEST".to_string());
    }

    #[test]
    fn test_char_unicode_handling() {
        let a = create_string_array(vec!["héllo", "wörld", "tëst"]);
        let result = upper(&a).unwrap();
        assert_eq!(result.get(0).unwrap(), &"HÉLLO".to_string());
        assert_eq!(result.get(1).unwrap(), &"WÖRLD".to_string());
        assert_eq!(result.get(2).unwrap(), &"TËST".to_string());
    }

    #[test]
    fn test_char_complex_strings() {
        let a = create_string_array(vec!["  hello_world  ", "\t\tTest\tCase\n\n"]);
        let result = strip(&a).unwrap();
        assert_eq!(result.get(0).unwrap(), &"hello_world".to_string());
        assert_eq!(result.get(1).unwrap(), &"Test\tCase".to_string());
    }
}
