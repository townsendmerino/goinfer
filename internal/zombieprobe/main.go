// Command zombieprobe prints the x86 features this host's CPUID and XCR0 report (temporary, zombie-probe branch only).
package main

import (
	"fmt"

	"golang.org/x/sys/cpu"
)

func main() {
	fmt.Printf("%+v\n", cpu.X86)
}
