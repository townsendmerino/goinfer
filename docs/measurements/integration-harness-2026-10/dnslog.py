#!/usr/bin/env python3
"""Log every DNS name looked up, answering NXDOMAIN at once: dnslog.py <logfile>. Run inside a network namespace whose
/etc/resolv.conf names 127.0.0.1, so the names a harness tries to reach are recorded and nothing resolves."""
import socket, struct, sys, time
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("127.0.0.1", 53))
log = open(sys.argv[1], "a", buffering=1)
while True:
    d, a = s.recvfrom(512)
    i, name = 12, []
    while d[i]:
        n = d[i]; name.append(d[i+1:i+1+n].decode("ascii", "replace")); i += n + 1
    qtype = struct.unpack(">H", d[i+1:i+3])[0]
    log.write(f"{time.time():.3f} {'.'.join(name)} type={qtype}\n")
    q = d[12:i+5]
    s.sendto(d[:2] + b"\x81\x83" + d[4:6] + b"\x00\x00\x00\x00\x00\x00" + q, a)
