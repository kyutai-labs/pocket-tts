fn main() {
    let cursor = std::io::Cursor::new(vec![]);
    let reader = hound::WavReader::new(cursor);
    match reader {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("Error Variant Debug: {:?}", e),
    }
}
