
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq)]
enum TestFileFormat {
    Npy,
    Npz,
    Text,
}

impl std::str::FromStr for TestFileFormat {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "npy" => Ok(TestFileFormat::Npy),
            "npz" => Ok(TestFileFormat::Npz),
            "txt" => Ok(TestFileFormat::Text),
            _ => Err(format!("Unknown format: {}", s)),
        }
    }
}

#[test]
fn test_format_detection_simple() {
    assert_eq!("npy".parse::<TestFileFormat>().unwrap(), TestFileFormat::Npy);
    assert_eq!("npz".parse::<TestFileFormat>().unwrap(), TestFileFormat::Npz);
    assert_eq!("txt".parse::<TestFileFormat>().unwrap(), TestFileFormat::Text);
}