// "abc": This is a string literal, which is a fixed-size sequence of characters. 
// String literals are stored in the program's binary and have a static lifetime. 
// They are usually used for short, known strings that don't need to be mutated.
// Ownership: Owned by the program itself (it's a part of the binary).

// String::from("abc"): This creates a String type instance by dynamically allocating 
// memory on the heap to store the string content. String is a dynamically resizable, 
// heap-allocated string type that can be modified and grown.
// Ownership: Owned by the heap, and a String value owns its content.

// In practice, you would use string literals when you know the content won't change, 
// and you want to optimize for memory usage and performance.