use ferris_says::say;
use std::io::{stdout, BufWriter};

fn print_hello_world() {
    println!("Hello, world!");
}

fn ferris_says() {
    let stdout = stdout();
    let message = String::from("Hello fellow Rustaceans!");
    let width = message.chars().count();

    let mut writer = BufWriter::new(stdout.lock());
    say(message.as_bytes(), width, &mut writer).unwrap();
}

fn main() {
    print_hello_world();
    ferris_says();
}
