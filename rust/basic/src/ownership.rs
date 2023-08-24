pub fn test_ownership1() {
    let a = String::from("hello");
    let b = a;
    // println!("a: {}", a); // error: value borrowed here after move
    // reason: the ownership of a is moved to b, 
    // so a is invalid, can't be used again in println!
}

pub fn test_ownership2() {
    let a = String::from("hello");
    let b = &a;
    println!("a: {}, b: {}", a, b);
    // this is valid because the ownership of a is not moved
    // something like a = b will move the ownership of a to b
    // but b = &a will not move the ownership of a
}