package main

import (
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"fmt"
)

func main() {
	dsn := "root:root@tcp(127.0.0.1:3306)/test?charset=utf8mb4&parseTime=True&loc=Local"
	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{})
	fmt.Println(db, err)
}