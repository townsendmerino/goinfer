- **Prebuilt WebGPU binaries.** Each release now carries `goinfer-serve-webgpu-linux-amd64`, `-linux-arm64` and
  `-windows-amd64.exe`, with their own `checksums-webgpu.txt` and build-provenance attestation, built with cgo and
  wgpu-native linked in. The static assets and `checksums.txt` are unchanged, and a WebGPU build failure cannot hold them
  back (#1177).
