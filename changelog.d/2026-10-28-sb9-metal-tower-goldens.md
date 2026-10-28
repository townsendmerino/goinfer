- **Per-tower goldens on Metal and the darwin CPU** (SB9, step S, tests only). The Metal backend's 17 tower cases (7
  families) and the arm64 CPU's 22 are now compared bit for bit with committed goldens on their fixed inputs. The Metal
  golden names the GPU and the macOS build, since Metal's compiler and driver ship with the OS, and a golden from another
  build fails until it is deliberately re-recorded. A planted one-ulp change in any tower turns exactly that tower red. On
  the M1 Pro every Metal tower agrees with the CPU at cosine 0.99999997 or better. No tower's output changed.
