use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;

pub struct Car {
    pub name: String,
    pub price: u32,
    pub owner: String,
}

impl Debug for Car {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_struct("Car");
        
        d.field("name", &self.name)
         .field("price", &self.price)
         .field("owner", &self.owner);
    
        d.finish_non_exhaustive()
    }
}