# Third-party licenses

> **This file should be regenerated as part of the release, not hand-maintained.**
> The authoritative version is the output of
> [`go-licenses`](https://github.com/google/go-licenses):
>
> ```
> go-licenses report ./... > THIRD_PARTY_LICENSES.md          # core
> go-licenses report -tags gpu ./gpu/... >> THIRD_PARTY_LICENSES.md   # WebGPU backend
> go-licenses report -tags cuda ./cuda/... >> THIRD_PARTY_LICENSES.md # CUDA backend
> ```
>
> That pulls the **exact** license text and copyright line from each module in the
> build. The entries below are an interim, hand-written scaffold so the required
> notices ship immediately — **confirm each copyright line against the module's own
> `LICENSE` file (which `go-licenses` does automatically) before the release.**
>
> `NOTICE` carries the human-readable summary; this file carries the full texts
> that MIT / Apache / BSD require to travel with the distribution.

---

## Core (default build)

- **golang.org/x/text** — BSD 3-Clause. Copyright (c) The Go Authors.
- **github.com/townsendmerino/aikit** — MIT. Copyright (c) 2026 Francis Townsend-Merino.

*(Full texts: generate via `go-licenses`.)*

---

## Optional WebGPU backend (`-tags gpu`, ./gpu)

- **github.com/cogentcore/webgpu** — BSD 3-Clause.

*(Full text: generate via `go-licenses`.)*

---

## Optional CUDA backend (`-tags cuda`, ./cuda)

goinfer's cgo-free CUDA backend is built on **gocudrv** by eitam ring. See the
release notes for the full credit and
<https://eitamring.github.io/posts/gocudrv-ten-weekends.html>.

### github.com/eitamring/gocudrv — MIT License

> Confirm the exact copyright line from the module's `LICENSE`. The permission
> text below is the standard MIT body (invariant).

```
MIT License

Copyright (c) eitam ring

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### github.com/ebitengine/purego — Apache License 2.0

*(gocudrv's dlopen/FFI dependency. **Full Apache-2.0 text + its `NOTICE` must be
included** — do not hand-copy; generate via `go-licenses`, which reproduces the
complete license and any upstream NOTICE. Copyright (c) The Ebitengine Authors.)*
