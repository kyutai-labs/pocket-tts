#[cfg(test)]
mod io_tests {
    use super::*;
    
    #[test]
    fn test_mmap_mode_parsing() {
        assert!("r".parse::<MmapMode>().unwrap() == MmapMode::Read);
        assert!("r+".parse::<MmapMode>().unwrap() == MmapMode::ReadWrite);
        assert!("c".parse::<MmapMode>().unwrap() == MmapMode::Read);
    }
    
    #[test]
    fn test_detect_file_format() {
        let result = detect_file_format_from_filename("test.npy");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), FileFormat::Npy);
        
        let result = detect_file_format_from_filename("test.npz");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), FileFormat::Npz);
        
        let result = detect_file_format_from_filename("test.txt");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), FileFormat::Text);
    }
}