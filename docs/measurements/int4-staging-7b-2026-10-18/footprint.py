#!/usr/bin/env python3
"""Prints a process's physical footprint now and its lifetime peak (darwin proc_pid_rusage, rusage_info_v4), in MB.
The lifetime peak counts from the process's start, so for a server read once it is up it covers the whole load.
Usage: footprint.py PID"""
import ctypes, sys

class RUsageV4(ctypes.Structure):  # <libproc.h> rusage_info_v4, through ri_lifetime_max_phys_footprint
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
        "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
        "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups", "ri_child_pageins",
        "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten", "ri_cpu_time_qos_default",
        "ri_cpu_time_qos_maintenance", "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
        "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive", "ri_billed_system_time",
        "ri_serviced_system_time", "ri_logical_writes", "ri_lifetime_max_phys_footprint")] + [("pad", ctypes.c_uint64 * 32)]

ru = RUsageV4()
if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(int(sys.argv[1]), 4, ctypes.byref(ru)) != 0:
    sys.exit("proc_pid_rusage failed")
print(f"phys_footprint {ru.ri_phys_footprint / 2**20:.0f} MB, lifetime_max_phys_footprint {ru.ri_lifetime_max_phys_footprint / 2**20:.0f} MB, resident {ru.ri_resident_size / 2**20:.0f} MB")
