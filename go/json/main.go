package main

import "fmt"
import "encoding/json"
import "reflect"

func main(){
	data := map[string]string{
		"foo": "bar",
		"baz": "qux",
	}

	// json.Marshal returns a byte array
	bytes_data, err := json.Marshal(data)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(bytes_data)
	fmt.Println(reflect.TypeOf(bytes_data))

	// json.Unmarshal returns an object, but you need to 
	// specify the pointer it will be assigned to
	var obj_data map[string]string
	err = json.Unmarshal(bytes_data, &obj_data)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(obj_data)
	fmt.Println(reflect.TypeOf(obj_data))
}