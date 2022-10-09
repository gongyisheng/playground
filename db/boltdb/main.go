package main

import (
	"log"
	"github.com/boltdb/bolt"
)

func main() {
	// create db
	db, err := bolt.Open("data/my.db", 0600, nil)
	if err != nil {
		panic(err)
	}
	defer db.Close()

	// insert data to db
	err = db.Update(func(tx *bolt.Tx) error {
		// create bucket
		bucket, err := tx.CreateBucketIfNotExists([]byte("user"))
		if err != nil {
			log.Fatalf("CreateBucketIfNotExists err:%s", err.Error())
			return err
		}
		// put data
		if err = bucket.Put([]byte("hello"), []byte("world")); err != nil {
			log.Fatalf("bucket Put err:%s", err.Error())
			return err
		}
		return nil
	})
	if err != nil {
		log.Fatalf("db.Update err:%s", err.Error())
	}
	// read data from db
	err = db.View(func(tx *bolt.Tx) error {
		// read bucket
		bucket := tx.Bucket([]byte("user"))
		// read data
		// correct key
		val := bucket.Get([]byte("hello"))
		log.Printf("the get val:%s", val)
		// wrong key
		val = bucket.Get([]byte("hello2"))
		log.Printf("the get val2:%s", val)
		return nil
	})
	if err != nil {
		log.Fatalf("db.View err:%s", err.Error())
	}
}