package main

import (
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"fmt"
)
type User struct {
	ID   uint   `gorm:"primaryKey"`
	Name string `gorm:"not null"`
	Age  int
}

func ConnectToDB(dsn string) (*gorm.DB, error) {
	var err error
	var db *gorm.DB

	db, err = gorm.Open(mysql.Open(dsn), &gorm.Config{})
	if err != nil {
		return nil, err
	}
	fmt.Println("Connection opened successfully")
	return db, nil
}

func CreateTable(db *gorm.DB) error {
	err := db.AutoMigrate(&User{})
	if err != nil {
		return err
	}
	fmt.Println("Table created successfully")
	return nil
}

func DropTable(db *gorm.DB) error {
	err := db.Migrator().DropTable(&User{})
	if err != nil {
		return err
	}
	fmt.Println("Table dropped successfully")
	return nil
}

func InsertUser(db *gorm.DB, user *User) error {
	result := db.Create(user)
	if result.Error != nil {
		return result.Error
	}
	fmt.Println("User inserted successfully")
	return nil
}

func GetUserByID(db *gorm.DB, id uint) (*User, error) {
	user := &User{}
	result := db.First(user, id)
	if result.Error != nil {
		return nil, result.Error
	}
	return user, nil
}

func UpdateUser(db *gorm.DB, user *User) error {
	result := db.Save(user)
	if result.Error != nil {
		return result.Error
	}
	fmt.Println("User updated successfully")
	return nil
}

func main() {
	dsn := "root:root@tcp(127.0.0.1:3306)/test?charset=utf8mb4&parseTime=True&loc=Local"
	db, err := ConnectToDB(dsn)
	if err != nil {
		panic(err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		panic(err)
	}
	defer sqlDB.Close()

	CreateTable(db)
	user1 := &User{ID: 1, Name: "John", Age: 20}
	user2 := &User{ID: 2, Name: "Tom", Age: 25}
	user3 := &User{ID: 3, Name: "Edward", Age: 30}
	InsertUser(db, user1)
	InsertUser(db, user2)
	InsertUser(db, user3)
	_user, err := GetUserByID(db, 1)
	if err != nil {
		panic(err)
	}
	fmt.Println("Get user:", _user.ID, _user.Name, _user.Age)
	_user.Age = 35
	UpdateUser(db, _user)
	_user, err = GetUserByID(db, 1)
	if err != nil {
		panic(err)
	}
	fmt.Println("Get user:", _user.ID, _user.Name, _user.Age)
	DropTable(db)
}
