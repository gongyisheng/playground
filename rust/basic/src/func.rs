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