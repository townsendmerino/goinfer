# goinfer — agent notes

## Models: stored remote, benchmarked local

Model weights live on the Linux box in `/srv/models` (5.5 TB), shared to the LAN as
`//192.168.1.240/models`. Each machine benchmarks from its **own local disk**, never
from the archive.

| machine | bench set (measure here) | archive access |
|---|---|---|
| `nobara-pc` (amd64/CUDA) | `~/models` on NVMe | `/srv/models` is local — no pull needed |
| MacBook (arm64/Metal) | `~/models` on internal SSD | `models-pull` from the archive |

**Never benchmark a path under `/Volumes/`.** That is the SMB mount; a run that reads
its checkpoint over the network measures the LAN and the server's 5400 rpm SMR disk
instead of the engine. It does not error — it returns a plausible, wrong number. Any
row whose model path starts with `/Volumes/` is void and must be re-measured.

### Getting a model on the MacBook

```sh
models-pull -l                          # list the archive
models-pull -l qwen                     # filtered
models-pull qwen3.6-35b-a3b-int4.giw    # archive -> ~/models, resumable
models-push my-new-download.gguf        # ~/models -> archive, size-verified
```

`models-push` refuses to report success unless byte counts match both sides — archive a
new download *before* deleting the local copy. Both scripts ride rsync over SSH and honour
`MODELS_HOST` (default `nobara`), `MODELS_ARCHIVE`, `MODELS_LOCAL`.

Preconditions: **must be on the LAN.** Tailscale SSH intercepts port 22 and will hang, so
`models-pull` does not work off-LAN. The box must be awake. Check free space first — the
35B Q8_0 is 37.8 GB.

## Benchmarking

`docs/benchmarks.md` is provenance-gated: a number enters a table only with machine,
checkpoint+quant, greedy/seed, pinned versions, date, thermal note, and local-disk path.
Read its Methodology section before adding or changing any measurement, and reproduce via
`scripts/bench_compare.sh`. Peer comparisons must be same-session interleaved — drift
between sessions is ~3.5% on this box and silently corrupts ratios.

CUDA rows are anchored to a specific NVIDIA driver version. Changing the driver invalidates
comparability and requires a deliberate re-anchor, not a silent carry-forward.
