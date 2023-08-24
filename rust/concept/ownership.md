Ownership is a fundamental concept in Rust that governs how memory is managed and shared.   
It ensures memory safety, prevents issues like data races and null pointer dereferences, and enforces predictable behavior at runtime.

In Rust, every value has a single owner at any given time. The owner is responsible for the memory cleanup when the value goes out of scope. Here are some key ownership rules:

Each value in Rust has a variable that's its owner.
There can only be one owner at a time.
When the owner goes out of scope, the value is automatically dropped and its memory is deallocated.
Let's look at ownership through examples:

```rust
fn main() {
    // Example 1: Ownership and Move
    let s1 = String::from("hello");
    let s2 = s1; // s1 is moved to s2, s1 can no longer be used

    // Example 2: Clone to Create a Copy
    let s3 = s2.clone(); // Creates a new copy, both s2 and s3 are valid

    // Example 3: Ownership and Functions
    let s4 = String::from("world");
    takes_ownership(s4); // s4 is moved into the function

    // Example 4: Returning Ownership
    let s5 = gives_ownership(); // Ownership is returned to s5

    // Example 5: References and Borrowing
    let s6 = String::from("example");
    let len = calculate_length(&s6); // Borrow s6's reference

    println!("{}", len); // len can be used here
}

fn takes_ownership(some_string: String) {
    println!("{}", some_string);
} // some_string goes out of scope and is dropped

fn gives_ownership() -> String {
    let s = String::from("returning ownership");
    s // s is returned and ownership moves
}

fn calculate_length(s: &String) -> usize {
    s.len()
} // s goes out of scope but no ownership is moved
```
In these examples:

Example 1: Ownership and Move  

s1 is the owner of the string "hello".  
s2 takes ownership of the string "hello" and s1 is invalidated.  

Example 2: Clone to Create a Copy  

s2 is cloned into s3, creating a new copy.  

Example 3: Ownership and Functions  

s4 is moved into the takes_ownership function, which takes ownership.  

Example 4: Returning Ownership  

Ownership of the string returned by gives_ownership is passed to s5.  

Example 5: References and Borrowing  

calculate_length borrows a reference to s6 without taking ownership.  

By enforcing these ownership rules, Rust ensures memory safety without requiring a garbage collector or manual memory management. It also encourages a clear understanding of data lifetimes and promotes efficient memory usage.

# Error Example
```rust
fn main() {
    let s = String::from("hello");
    let s1 = s;
    println!("{}", s); // Error: s is no longer valid
}
```

