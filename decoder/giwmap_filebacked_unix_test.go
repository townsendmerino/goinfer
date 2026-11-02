//go:build unix || windows

package decoder

// giwMapIsFileBacked: a .giw here is a read-only file mapping, so an aliased weight is file-backed. See the other file.
const giwMapIsFileBacked = true
