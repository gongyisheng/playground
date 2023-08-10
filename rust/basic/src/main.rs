mod container;
mod enumerate;
mod func;
mod oop;

fn main() {

    // func test
    println!("-----func test-----");

    func::ferris_says();

    println!("-------------------");
    // same thing, different import
    func::greet("world");
    crate::func::greet("world");
    
    println!("-------------------");
    
    println!("fibonacci(10): {}", func::fibonacci(10));
    
    println!("-------------------");

    let s1 = String::from("Hello");
    let s2 = String::from("World !");
    println!("get_longer_str(\"{}\", \"{}\") = \"{}\"", s1, s2, func::get_longer_str(&s1, &s2));

    println!("-------------------");

    let mut s = String::from("Hello");
    println!("s before modify: {}", s);
    func::modify(&mut s);
    println!("s after modify: {}", s);

    println!("-------------------");

    let result = func::divide(10.0, 2.0);
    match result {
        Ok(value) => println!("Divide Result: {}", value),
        Err(error) => println!("Divide Error: {}", error),
    }
    let result = func::divide(10.0, 0.0);
    match result {
        Ok(value) => println!("Divide Result: {}", value),
        Err(error) => println!("Divide Error: {}", error),
    }

    println!("-------------------");
    
    func::variable_test();
    
    println!("-------------------");

    // enum test
    println!("-----enum test-----");

    let coin = enumerate::Coin::Penny;
    let cents = enumerate::value_in_cents(coin);
    println!("Value: {} cents", cents);

    let coin = enumerate::Coin::Nickel;
    let cents = enumerate::value_in_cents(coin);
    println!("Value: {} cents", cents);

    let coin = enumerate::Coin::Dime;
    let cents = enumerate::value_in_cents(coin);
    println!("Value: {} cents", cents);

    let coin = enumerate::Coin::Quarter(enumerate::UsState::Alaska);
    let cents = enumerate::value_in_cents(coin);
    println!("Value: {} cents", cents);

    let coin = enumerate::Coin::Quarter(enumerate::UsState::Alabama);
    let cents = enumerate::value_in_cents(coin);
    println!("Value: {} cents", cents);

    // container test
    println!("---container test---");

    container::create_list1();
    container::create_list2();
    container::access_list();
    container::delete_list();
    container::iter_list();

    println!("-------------------");

    container::create_hashmap(true);
    container::access_hashmap();
    container::add_hashmap();
    container::update_hashmap();
    container::delete_hashmap();
    container::iter_hashmap();

    // oop test
    println!("-----oop test-----");
    let mut rec = oop::Rectangle { width: 30, height: 50 };
    println!("rec before update: {:?}, {:?}", rec.width, rec.height);
    println!("rec area: {:?}", rec.area());

    rec.set_width(10);
    rec.set_height(20);
    println!("rec after update: {:?}, {:?}", rec.width, rec.height);
    println!("rec area: {:?}", rec.area());

    let tiny_rec = oop::Rectangle { width: 5, height: 5 };
    println!("tiny rec: {:?}, {:?}", tiny_rec.width, tiny_rec.height);
    println!("rec can hold tiny rec: {:?}", rec.can_hold(&tiny_rec));
}
