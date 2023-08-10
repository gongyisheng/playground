
pub fn create_list1(){
    let mut my_list: Vec<i32> = Vec::new();
    my_list.push(10);
    my_list.push(20);
    my_list.push(30);
    println!("[create1] my_list: {:?}", my_list);
}

pub fn create_list2(){
    let my_list = vec![10, 20, 30];
    println!("[create2] my_list: {:?}", my_list);
}

pub fn access_list(){
    let my_list = vec![10, 20, 30];
    println!("[access] my_list[0]: {}", my_list[0]);
    println!("[access] my_list[1]: {}", my_list[1]);
    println!("[access] my_list[2]: {}", my_list[2]);
}

pub fn delete_list(){
    let mut my_list = vec![10, 20, 30];
    println!("[delete] my_list: {:?}", my_list);
    my_list.remove(1);
    println!("[delete] my_list: {:?}", my_list);
    my_list.pop();
    println!("[delete] my_list: {:?}", my_list);
}

pub fn iter_list(){
    let names = ["Alice", "Bob", "Charlie"];
    for name in names.iter() {
        println!("[iter] Hello {}!", name);
    }
}