//go:build !darwin && !linux

package probe

func read() (Reading, bool) { return Reading{}, false }

func readCPU() (CPUTicks, bool) { return CPUTicks{}, false }

func readTask(int) (TaskVM, bool) { return TaskVM{}, false }

func readWithOracle() (OracleSample, bool) { return OracleSample{}, false }
