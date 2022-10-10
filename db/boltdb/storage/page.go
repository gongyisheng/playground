package storage

type pgid uint64

type page struct {
	// pgid is of type pgid and is the number given to the page. 8 bytes
    id pgid
	// flags is the specific type of data saved in this page. (There are several) 2 bytes
    flags uint16
	// count records the count in a specific data type, different types have different meanings. 2 bytes
    count uint16
	// overflow used to record if there is an inter-page. 4 bytes
    overflow uint32
	// ptr is the concrete data type
    ptr uintptr
}

// data struct in disk: <id>-<flags>-<count>-<overflow>-<ptr>

// There are 4 different kinds of pages
const (
	branchPageFlag   = 0x01 // branch node in b+ tree, serving as an index node, store pgid and key.
	leafPageFlag     = 0x02 // leaf node in b+ tree, serving as a storage node, store pgid, key and value.
	metaPageFlag     = 0x04 // store metadata of db, eg. freelistpage id, bucket root page
	freelistPageFlag = 0x10 // store the pgid of free page, new data will be written to free page first.
)
