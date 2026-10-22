//go:build darwin

#include "textflag.h"

// One trampoline per libSystem function libsystem_darwin.go imports; the same text assembles for amd64 and arm64.

TEXT libc_mach_host_self_trampoline<>(SB),NOSPLIT,$0-0
	JMP	libc_mach_host_self(SB)
GLOBL	·libc_mach_host_self_trampoline_addr(SB), RODATA, $8
DATA	·libc_mach_host_self_trampoline_addr(SB)/8, $libc_mach_host_self_trampoline<>(SB)

TEXT libc_mach_task_self_trampoline<>(SB),NOSPLIT,$0-0
	JMP	libc_mach_task_self(SB)
GLOBL	·libc_mach_task_self_trampoline_addr(SB), RODATA, $8
DATA	·libc_mach_task_self_trampoline_addr(SB)/8, $libc_mach_task_self_trampoline<>(SB)

TEXT libc_host_statistics64_trampoline<>(SB),NOSPLIT,$0-0
	JMP	libc_host_statistics64(SB)
GLOBL	·libc_host_statistics64_trampoline_addr(SB), RODATA, $8
DATA	·libc_host_statistics64_trampoline_addr(SB)/8, $libc_host_statistics64_trampoline<>(SB)

TEXT libc_host_page_size_trampoline<>(SB),NOSPLIT,$0-0
	JMP	libc_host_page_size(SB)
GLOBL	·libc_host_page_size_trampoline_addr(SB), RODATA, $8
DATA	·libc_host_page_size_trampoline_addr(SB)/8, $libc_host_page_size_trampoline<>(SB)

TEXT libc_task_name_for_pid_trampoline<>(SB),NOSPLIT,$0-0
	JMP	libc_task_name_for_pid(SB)
GLOBL	·libc_task_name_for_pid_trampoline_addr(SB), RODATA, $8
DATA	·libc_task_name_for_pid_trampoline_addr(SB)/8, $libc_task_name_for_pid_trampoline<>(SB)

TEXT libc_task_info_trampoline<>(SB),NOSPLIT,$0-0
	JMP	libc_task_info(SB)
GLOBL	·libc_task_info_trampoline_addr(SB), RODATA, $8
DATA	·libc_task_info_trampoline_addr(SB)/8, $libc_task_info_trampoline<>(SB)
