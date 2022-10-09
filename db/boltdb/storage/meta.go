// meta page 
// struct - db.go
type meta struct {
    magic    uint32 // magic number
    version  uint32 // version
    pageSize uint32 // size of page, the value should be same as default page size of operation system.
    flags    uint32 // reserved value, not used now.
    root     bucket // root bucket
    freelist pgid // id of free list page
    pgid     pgid // id of this page
    txid     txid // max transaction id
    checksum uint64 // for checking purpose
}

// struct in disk: <magic>-<version>-<pagesize>-<flags>-<root(bucket)>-<freelist_id>-<pgid>-<txid>-<checksum>

// meta function - page.go
// meta returns a pointer to the metadata section of the page.
func (p *page) meta() *meta {
    // convert p.ptr to meta
    return (*meta)(unsafe.Pointer(&p.ptr))
}

// write
// write the meta onto a page - db.go
func (m *meta) write(p *page) {
    if m.root.root >= m.pgid {
        panic(fmt.Sprintf("root bucket pgid (%d) above high water mark (%d)", m.root.root, m.pgid))
    } else if m.freelist >= m.pgid {
        panic(fmt.Sprintf("freelist pgid (%d) above high water mark (%d)", m.freelist, m.pgid))
    }
    // Page id is either going to be 0 or 1 which we can determine by the transaction ID.
    p.id = pgid(m.txid % 2)
    p.flags |= metaPageFlag
    // Calculate the checksum.
    m.checksum = m.sum64()
	// p.meta() returns the address of p.ptr. meta data is written to database after copy()
    m.copy(p.meta())
}

// copy copies one meta object to another.
func (m *meta) copy(dest *meta) {
    *dest = *m
}
// generates the checksum for the meta.
func (m *meta) sum64() uint64 {
    var h = fnv.New64a()
    _, _ = h.Write((*[unsafe.Offsetof(meta{}.checksum)]byte)(unsafe.Pointer(m))[:])
    return h.Sum64()
}

// read
// read data from a page - page.go
// meta returns a pointer to the metadata section of the page.
func (p *page) meta() *meta {
	// convert p.ptr to meta
    return (*meta)(unsafe.Pointer(&p.ptr))
}