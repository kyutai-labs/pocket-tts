// Simple test to verify bitwise module works
use rust_numpy::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Test bitwise AND
    let a = array![5u8, 3u8, 7u8];
    let b = array![2u8, 6u8, 1u8];
    let result = a.bitwise_and(&b)?;
    println!("bitwise_and result: {:?}", result.to_vec());

    // Test bitwise OR
    let result_or = a.bitwise_or(&b)?;
    println!("bitwise_or result: {:?}", result_or.to_vec());

    // Test bitwise XOR
    let result_xor = a.bitwise_xor(&b)?;
    println!("bitwise_xor result: {:?}", result_xor.to_vec());

    // Test bitwise NOT
    let result_not = a.bitwise_not()?;
    println!("bitwise_not result: {:?}", result_not.to_vec());

    Ok(())
}
