use numpy::array::Array;
use numpy::math_ufuncs::unwrap;

fn main() {
    // Test basic unwrap
    let p: Array<f64> = Array::from_data(vec![5.5, 5.8, 6.1, 0.2, 0.5], vec![5]);
    let result = unwrap(&p, None, None, None).unwrap();

    println!("Test 1D basic:");
    for i in 0..result.size() {
        let val = *result.get(i).unwrap();
        println!("  [{}] = {}", i, val);
    }

    // Test 2D
    let p2: Array<f64> = Array::from_data(vec![0.0, 0.5, 6.0, 6.5, 1.0, 1.5, 1.8, 2.0], vec![2, 4]);
    let result2 = unwrap(&p2, None, None, None).unwrap();

    println!("\nTest 2D:");
    for i in 0..result2.size() {
        let val = *result2.get(i).unwrap();
        println!("  [{}] = {}", i, val);
    }
}
