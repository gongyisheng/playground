package main

import (
	"fmt"
)

func update_map(mp *map[string]interface{}) {
	new_map := make(map[string]interface{})
	new_map["new_key"] = "new_value"
	*mp = new_map
}

func main() {
	old_map := make(map[string]interface{})
	old_map["old_key"] = "old_value"
	update_map(&old_map)
	fmt.Println(old_map)
}