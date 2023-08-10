#[derive(Debug)]
pub enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter(UsState), // Quarter is a variant of the Coin enum, and it takes an associated value of the UsState enum
}

#[derive(Debug)]
pub enum UsState {
    Alabama,
    Alaska,
    // ... other states
}

pub fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => 1,
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter(state) => {
            println!("State quarter from {:?}", state);
            25
        },
    }
}
