use ferris_says::say;
use std::io::{stdout, BufWriter};

pub fn greet(name: &str) {
    println!("Hello, {}!", name);
}

pub fn ferris_says() {
    let stdout = stdout();
    let message = String::from("Hello fellow Rustaceans!");
    let width = message.chars().count();

    let mut writer = BufWriter::new(stdout.lock());
    say(message.as_bytes(), width, &mut writer).unwrap();
}

pub fn fibonacci(n: u64) -> u64 {
    if n == 0 {
        0
    } else if n == 1 {
        1
    } else {
        fibonacci(n - 1) + fibonacci(n - 2)
    }
}

pub fn variable_test() {
    let mut counter = 0;          // Declare a mutable variable
    let value = counter;          // Copy value, no ownership transfer
    counter += 1;                 // Modify the mutable variable
    println!("Counter: {}, Value: {}", counter, value);
}

// Lifetime Annotations
pub fn get_longer_str<'a>(s1: &'a str, s2: &'a str) -> &'a str {
    if s1.len() > s2.len() {
        s1
    } else {
        s2
    }
}

// Borrowing and References
pub fn modify(s: &mut String) {
    s.push_str(" World");
}

// Option and Result Enums
pub fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b != 0.0 {
        Ok(a / b)
    } else {
        Err("Cannot divide by zero".to_string())
    }
}
