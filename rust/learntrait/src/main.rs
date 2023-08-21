mod learndebug;

fn main() {
    let car = learndebug::Car {
        name: String::from("BMW"),
        price: 1000000,
        owner: String::from("John"),
    };
    println!("car: {:?}", car);
}
