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

    // container test
    println!("---container test---");

    container::create_list1();
    container::create_list2();
    container::access_list();
    container::delete_list();
    container::iter_list(); 
}
