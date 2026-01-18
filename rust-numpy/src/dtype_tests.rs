//! Dtype unit tests
//! Tests for dtype system including string parsing, type inference, and special values

use crate::dtype::{Dtype, DtypeKind};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex32_parsing() {
        // Test that complex32 can be parsed
        let result = Dtype::from_str("complex32");
        assert!(result.is_ok(), "Failed to parse 'complex32'");
        match result.unwrap() {
            Dtype::Complex32 { .. } => (),
            _ => panic!("Parsed as wrong dtype"),
        }
    }

    #[test]
    fn test_complex32_c4_parsing() {
        // Test that c4 shorthand parses to complex32
        let result = Dtype::from_str("c4");
        assert!(result.is_ok(), "Failed to parse 'c4'");
        match result.unwrap() {
            Dtype::Complex32 { .. } => (),
            _ => panic!("Parsed as wrong dtype"),
        }
    }

    #[test]
    fn test_intp_parsing() {
        // Test that intp can be parsed
        let result = Dtype::from_str("intp");
        assert!(result.is_ok(), "Failed to parse 'intp'");
        match result.unwrap() {
            Dtype::Intp { .. } => (),
            _ => panic!("Parsed as wrong dtype"),
        }
    }

    #[test]
    fn test_intp_ip_parsing() {
        // Test that ip shorthand parses to intp
        let result = Dtype::from_str("ip");
        assert!(result.is_ok(), "Failed to parse 'ip'");
        match result.unwrap() {
            Dtype::Intp { .. } => (),
            _ => panic!("Parsed as wrong dtype"),
        }
    }

    #[test]
    fn test_uintp_parsing() {
        // Test that uintp can be parsed
        let result = Dtype::from_str("uintp");
        assert!(result.is_ok(), "Failed to parse 'uintp'");
        match result.unwrap() {
            Dtype::Uintp { .. } => (),
            _ => panic!("Parsed as wrong dtype"),
        }
    }

    #[test]
    fn test_uintp_up_parsing() {
        // Test that up shorthand parses to uintp
        let result = Dtype::from_str("up");
        assert!(result.is_ok(), "Failed to parse 'up'");
        match result.unwrap() {
            Dtype::Uintp { .. } => (),
            _ => panic!("Parsed as wrong dtype"),
        }
    }

    #[test]
    fn test_intp_from_type_isize() {
        // Test that isize maps to Intp
        let dtype = Dtype::from_type::<isize>();
        match dtype {
            Dtype::Intp { .. } => (),
            _ => panic!("isize did not map to Intp"),
        }
    }

    #[test]
    fn test_uintp_from_type_usize() {
        // Test that usize maps to Uintp
        let dtype = Dtype::from_type::<usize>();
        match dtype {
            Dtype::Uintp { .. } => (),
            _ => panic!("usize did not map to Uintp"),
        }
    }

    #[test]
    fn test_intp_itemsize() {
        // Test that Intp has correct itemsize
        let dtype = Dtype::Intp { byteorder: None };
        let itemsize = dtype.itemsize();
        #[cfg(target_pointer_width = "64")]
        assert_eq!(itemsize, 8, "Intp itemsize incorrect on 64-bit");
        #[cfg(target_pointer_width = "32")]
        assert_eq!(itemsize, 4, "Intp itemsize incorrect on 32-bit");
    }

    #[test]
    fn test_uintp_itemsize() {
        // Test that Uintp has correct itemsize
        let dtype = Dtype::Uintp { byteorder: None };
        let itemsize = dtype.itemsize();
        #[cfg(target_pointer_width = "64")]
        assert_eq!(itemsize, 8, "Uintp itemsize incorrect on 64-bit");
        #[cfg(target_pointer_width = "32")]
        assert_eq!(itemsize, 4, "Uintp itemsize incorrect on 32-bit");
    }

    #[test]
    fn test_intp_kind() {
        // Test that Intp has correct kind
        let dtype = Dtype::Intp { byteorder: None };
        let kind = dtype.kind();
        assert_eq!(kind, DtypeKind::Integer, "Intp kind incorrect");
    }

    #[test]
    fn test_uintp_kind() {
        // Test that Uintp has correct kind
        let dtype = Dtype::Uintp { byteorder: None };
        let kind = dtype.kind();
        assert_eq!(kind, DtypeKind::Unsigned, "Uintp kind incorrect");
    }

    #[test]
    fn test_f16_half_crate() {
        // Test that f16 is now using IEEE 754 compliant half crate
        use half::f16;

        let half = f16::from_f32(1.5);
        let back = half.to_f32();

        // Test basic conversion
        assert!((back - 1.5).abs() < f16::EPSILON, "f16 conversion failed");

        // Test special values
        let inf = f16::INFINITY;
        assert!(inf.to_f32().is_infinite(), "f16 infinity not preserved");

        let neg_inf = f16::NEG_INFINITY;
        assert!(neg_inf.to_f32().is_infinite(), "f16 negative infinity not preserved");

        let nan = f16::NAN;
        assert!(nan.to_f32().is_nan(), "f16 NaN not preserved");
    }

    #[test]
    fn test_float16_string_representation() {
        // Test that float16 still parses correctly
        let result = Dtype::from_str("float16");
        assert!(result.is_ok(), "Failed to parse 'float16'");
        match result.unwrap() {
            Dtype::Float16 { .. } => (),
            _ => panic!("Parsed as wrong dtype"),
        }
    }

    #[test]
    fn test_f2_string_representation() {
        // Test that f2 shorthand still parses to float16
        let result = Dtype::from_str("f2");
        assert!(result.is_ok(), "Failed to parse 'f2'");
        match result.unwrap() {
            Dtype::Float16 { .. } => (),
            _ => panic!("Parsed as wrong dtype"),
        }
    }

    #[test]
    fn test_all_dtypes_parse() {
        // Test that all major dtypes can be parsed
        let dtypes = [
            "int8", "i1", "int16", "i2", "int32", "i4", "int64", "i8",
            "intp", "ip",
            "uint8", "u1", "uint16", "u2", "uint32", "u4", "uint64", "u8",
            "uintp", "up",
            "float16", "f2", "float32", "f4", "float64", "f8",
            "complex32", "c4", "complex64", "c8", "complex128", "c16",
            "bool", "str", "unicode", "object",
        ];

        for dtype_str in dtypes.iter() {
            let result = Dtype::from_str(dtype_str);
            assert!(
                result.is_ok(),
                "Failed to parse dtype: {}",
                dtype_str
            );
        }
    }

    #[test]
    fn test_complex32_string_representation() {
        // Test that Complex32 converts to correct string
        let dtype = Dtype::Complex32 { byteorder: None };
        let s = dtype.to_string();
        assert_eq!(s, "complex32", "Complex32 string representation incorrect");
    }

    #[test]
    fn test_complex64_string_representation() {
        // Test that Complex64 still converts correctly
        let dtype = Dtype::Complex64 { byteorder: None };
        let s = dtype.to_string();
        assert_eq!(s, "complex64", "Complex64 string representation incorrect");
    }
}
