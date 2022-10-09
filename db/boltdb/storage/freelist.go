// freelist page

// freelist represents a list of all pages that are available for allocation.
// It also tracks pages that have been freed but are still in use by open transactions.

// struct - freelist.go
type freelist struct {
    ids     []pgid          // all free and available free page ids. (memory and disk)
    pending map[txid][]pgid // mapping of soon-to-be free page ids by transaction. (memory only)
    cache   map[pgid]bool   // fast lookup of all free and pending page ids. (memory only)
}
// If the len(ids)>0xFFFF, count will be 0xFFFF

// struct in disk: <pgidx or len(pgids)>-<pgidx>

// newFreelist returns an empty, initialized freelist.
func newFreelist() *freelist {
    return &freelist{
        pending: make(map[txid][]pgid),
        cache:   make(map[pgid]bool),
    }
}

// write
// write the page ids onto a freelist page. All free and pending ids are
// saved to disk since in the event of a program crash, all pending ids will
// become free - freelist.go
func (f *freelist) write(p *page) error {
    // Combine the old free pgids and pgids waiting on an open transaction.
    // Update the header flag.
    p.flags |= freelistPageFlag
    // The page.count can only hold up to 64k elements so if we overflow that
    // number then we handle it by putting the size in the first element.
    lenids := f.count()
    if lenids == 0 {
        p.count = uint16(lenids)
    } else if lenids < 0xFFFF {
        p.count = uint16(lenids)
		// copy to page.ptr
        f.copyall(((*[maxAllocSize]pgid)(unsafe.Pointer(&p.ptr)))[:])
    } else {
		// if there's overflow, put the first element as the length of ids
        p.count = 0xFFFF
        ((*[maxAllocSize]pgid)(unsafe.Pointer(&p.ptr)))[0] = pgid(lenids)
        // copy to page.ptr
        f.copyall(((*[maxAllocSize]pgid)(unsafe.Pointer(&p.ptr)))[1:])
    }
    return nil
}
// copyall copies into dst a list of all free ids and all pending ids in one sorted list.
// f.count returns the minimum length required for dst.
func (f *freelist) copyall(dst []pgid) {
    // put pending ids to a list
    m := make(pgids, 0, f.pending_count())
    for _, list := range f.pending {
        m = append(m, list...)
    }
	// sort the list
    sort.Sort(m)
    // merge the two sorted list. the final result is stored in dst
    mergepgids(dst, f.ids, m)
}
// mergepgids copies the sorted union of a and b into dst.
// If dst is too small, it panics.
func mergepgids(dst, a, b pgids) {
    if len(dst) < len(a)+len(b) {
        panic(fmt.Errorf("mergepgids bad len %d < %d + %d", len(dst), len(a), len(b)))
    }
    // Copy in the opposite slice if one is nil.
    if len(a) == 0 {
        copy(dst, b)
        return
    }
    if len(b) == 0 {
        copy(dst, a)
        return
    }
    // Merged will hold all elements from both lists.
    merged := dst[:0]
    // Assign lead to the slice with a lower starting value, follow to the higher value.
    lead, follow := a, b
    if b[0] < a[0] {
        lead, follow = b, a
    }
    // Continue while there are elements in the lead.
    for len(lead) > 0 {
        // Merge largest prefix of lead that is ahead of follow[0].
        n := sort.Search(len(lead), func(i int) bool { return lead[i] > follow[0] })
        merged = append(merged, lead[:n]...)
        if n >= len(lead) {
            break
        }
        // Swap lead and follow.
        lead, follow = follow, lead[n:]
    }
    // Append what's left in follow.
    _ = append(merged, follow...)
}

// read
// read initializes the freelist from a freelist page - freelist.go
func (f *freelist) read(p *page) {
    // If the page.count is at the max uint16 value (64k) then it's considered
    // an overflow and the size of the freelist is stored as the first element.
    idx, count := 0, int(p.count)
    if count == 0xFFFF {
        idx = 1
        // use the value of first uint64 number as count
        count = int(((*[maxAllocSize]pgid)(unsafe.Pointer(&p.ptr)))[0])
    }
    // Copy the list of page ids from the freelist.
    if count == 0 {
        f.ids = nil
    } else {
        ids := ((*[maxAllocSize]pgid)(unsafe.Pointer(&p.ptr)))[idx:count]
        f.ids = make([]pgid, len(ids))
        copy(f.ids, ids)
        // Make sure they're sorted.
        sort.Sort(pgids(f.ids))
    }
    // Rebuild the page cache.
    f.reindex()
}
// reindex rebuilds the free cache based on available and pending free lists.
func (f *freelist) reindex() {
    f.cache = make(map[pgid]bool, len(f.ids))
    for _, id := range f.ids {
        f.cache[id] = true
    }
    for _, pendingIDs := range f.pending {
        for _, pendingID := range pendingIDs {
            f.cache[pendingID] = true
        }
    }
}