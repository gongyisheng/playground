use std::collections::HashMap;

pub fn create_list1(){
    let mut my_list: Vec<i32> = Vec::new();
    my_list.push(10);
    my_list.push(20);
    my_list.push(30);
    println!("[list][create1] my_list: {:?}", my_list);
}

pub fn create_list2(){
    let my_list = vec![10, 20, 30];
    println!("[list][create2] my_list: {:?}", my_list);
}

pub fn access_list(){
    let my_list = vec![10, 20, 30];
    println!("[list][access] my_list[0]: {}", my_list[0]);
    println!("[list][access] my_list[1]: {}", my_list[1]);
    println!("[list][access] my_list[2]: {}", my_list[2]);
}

pub fn delete_list(){
    let mut my_list = vec![10, 20, 30];
    println!("[list][delete] my_list: {:?}", my_list);
    my_list.remove(1);
    println!("[list][delete] my_list: {:?}", my_list);
    my_list.pop();
    println!("[list][delete] my_list: {:?}", my_list);
}

pub fn iter_list(){
    let names = ["Alice", "Bob", "Charlie"];
    for name in names.iter() {
        println!("[list][iter] Hello {}!", name);
    }
}

pub fn create_hashmap(debug: bool) -> HashMap<String, i32> {
    let mut my_map: HashMap<String, i32> = HashMap::new();
    my_map.insert(String::from("one"), 1);
    my_map.insert(String::from("two"), 2);
    my_map.insert(String::from("three"), 3);
    if debug {
        println!("[hashmap][create1] my_map: {:?}", my_map);
    }
    return my_map;
}

pub fn access_hashmap(){
    let my_map = create_hashmap(false);
    println!("[hashmap][access] my_map[\"one\"]: {}", my_map["one"]);
    let key_list = ["two", "three", "four"];
    for _key in key_list.iter(){
        // map.get returns Option<&V>
        // Option is an enum, it can be either Some(v) or None
        if let Some(value) = my_map.get("one") {
            println!("[hashmap][access] my_map.get(\"one\"): {}", value);
        } else {
            println!("[hashmap][access] my_map.get(\"one\"): None");
        }
    }
}

pub fn add_hashmap(){
    let mut my_map = create_hashmap(false);
    println!("[hashmap][add] my_map: {:?}", my_map);
    my_map.insert(String::from("four"), 4);
    println!("[hashmap][add] my_map: {:?}", my_map);
}

pub fn update_hashmap(){
    let mut my_map = create_hashmap(false);
    println!("[hashmap][update] my_map: {:?}", my_map);
    my_map.insert(String::from("two"), 22);
    println!("[hashmap][update] my_map: {:?}", my_map);
}

pub fn delete_hashmap(){
    let mut my_map = create_hashmap(false);
    println!("[hashmap][delete] my_map: {:?}", my_map);
    my_map.remove("two");
    println!("[hashmap][delete] my_map: {:?}", my_map);
    my_map.clear();
    println!("[hashmap][delete] my_map: {:?}", my_map);
}

pub fn iter_hashmap(){
    let my_map = create_hashmap(false);
    for (key, value) in &my_map {
        println!("[hashmap][iter] key: {}, value: {}", key, value);
    }
}