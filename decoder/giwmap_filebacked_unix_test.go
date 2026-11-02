//go:build unix

package decoder

// giwMapIsFileBacked: a .giw here is a read-only file mapping, so an aliased weight is file-backed. See the non-unix file.
const giwMapIsFileBacked = true
