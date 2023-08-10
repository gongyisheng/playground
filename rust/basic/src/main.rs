mod container;
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
    
    func::variable_test();
    
    println!("-------------------");

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
