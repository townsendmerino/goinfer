//go:build !darwin && !linux

package hostmem

func read() (Reading, bool) { return Reading{}, false }

func readTask(int) (TaskVM, bool) { return TaskVM{}, false }
