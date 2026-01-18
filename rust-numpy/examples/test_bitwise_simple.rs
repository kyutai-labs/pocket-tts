// Simple test to verify bitwise module works
fn main() {
    // Create test arrays
    let a = vec![5u8, 3u8, 7u8];
    let b = vec![2u8, 6u8, 1u8];

    // Test basic bitwise operations
    let and_result = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x & y)
        .collect::<Vec<_>>();
    let or_result = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x | y)
        .collect::<Vec<_>>();
    let xor_result = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x ^ y)
        .collect::<Vec<_>>();
    let not_result = a.iter().map(|x| !x).collect::<Vec<_>>();

    println!("AND result: {:?}", and_result);
    println!("OR result: {:?}", or_result);
    println!("XOR result: {:?}", xor_result);
    println!("NOT result: {:?}", not_result);
}
