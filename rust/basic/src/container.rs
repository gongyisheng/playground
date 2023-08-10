pub fn iter_list(){
    let names = ["Alice", "Bob", "Charlie"];
    for name in names.iter() {
        println!("Hello {}!", name);
    }
}