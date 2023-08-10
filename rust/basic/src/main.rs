mod container;
mod func;


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

    // container test
    println!("---container test---");

    container::create_list1();
    container::create_list2();
    container::access_list();
    container::delete_list();
    container::iter_list();

    container::create_hashmap(true);
    container::access_hashmap();
    container::add_hashmap();
    container::update_hashmap();
    container::delete_hashmap();
    container::iter_hashmap();
}
