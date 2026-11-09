- **`--version` names the binary without `.exe` on Windows.** The first line of `goinfer-serve --version` and `goinfer-chat
  --version` read `goinfer-serve.exe v0.24.1` on Windows and `goinfer-serve v0.24.1` elsewhere; it is now `goinfer-serve <version>`
  on every platform, followed by ` (<commit>)` when the version does not already carry it, so a client that compares the version
  reads one form.
