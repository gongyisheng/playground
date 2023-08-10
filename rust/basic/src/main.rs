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

    container::iter_list();   
}
