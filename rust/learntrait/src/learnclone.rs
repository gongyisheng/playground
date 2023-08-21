use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;

pub struct Person {
    name: String,
    age: u32,
}

impl Clone for Person {
    fn clone(&self) -> Self {
        Person {
            name: self.name.clone(),
            age: self.age,
        }
    }
}

impl Debug for Person {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_struct("Person");
        
        d.field("name", &self.name)
         .field("age", &self.age);
    
        d.finish_non_exhaustive()
    }
}

pub fn demo_clone() {
    let p1 = Person {
        name: String::from("John"),
        age: 20,
    };
    let p2 = p1.clone();
    println!("p1: {:?}", p1);
    println!("p2: {:?}", p2);
}