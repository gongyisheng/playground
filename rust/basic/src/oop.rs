pub struct Rectangle {
    pub width: u32,
    pub height: u32,
}

// It is recommended to return -> &mut Self instead of nothing
// because it returns the instance of the struct on which the
// method is called, which allows us to chain methods

impl Rectangle {
    pub fn area(&self) -> u32 {
        self.width * self.height
    }

    pub fn set_width(&mut self, width: u32) -> &mut Self {
        // &mut self indicates that the method is allowed to modify 
        // the internal state of the instance of the struct on which 
        // the method is called
        self.width = width;

        self
    }

    pub fn set_height(&mut self, height: u32) -> &mut Self {
        // &mut self indicates that the method is allowed to modify 
        // the internal state of the instance of the struct on which 
        // the method is called
        self.height = height;

        self
    }

    pub fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.height > other.height
    }
}