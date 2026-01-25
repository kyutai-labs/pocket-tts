use numpy::array::Array;

fn main() {
    let a = Array::from_vec(vec![1i32, 2, 3, 4])
        .reshape(&[2, 2])
        .unwrap();
    let b = Array::from_vec(vec![1i32]);

    println!("a.shape(): {:?}", a.shape());
    println!("a.to_vec(): {:?}", a.to_vec());
    println!("b.shape(): {:?}", b.shape());
    println!("b.to_vec(): {:?}", b.to_vec());

    // Test manual broadcasting
    let broadcasted_b = b.broadcast_to(&[2, 2]).unwrap();
    println!("broadcasted_b.shape(): {:?}", broadcasted_b.shape());
    println!("broadcasted_b.to_vec(): {:?}", broadcasted_b.to_vec());

    // Test element-wise comparison manually
    let a_flat = a.to_vec();
    let b_flat = broadcasted_b.to_vec();
    let mut result = Vec::new();

    for (i, &a_val) in a_flat.iter().enumerate() {
        let cmp = a_val > b_flat[i];
        result.push(cmp);
        println!(
            "a[{}] = {}, b[{}] = {}, a > b = {}",
            i, a_val, i, b_flat[i], cmp
        );
    }

    println!("Manual result: {:?}", result);

    // Test using the greater method
    let greater_result = a.greater(&b).unwrap();
    println!("greater_result.shape(): {:?}", greater_result.shape());
    println!("greater_result.to_vec(): {:?}", greater_result.to_vec());
}
