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