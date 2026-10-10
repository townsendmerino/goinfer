# Night run 2026-10-09 — macbookpro.lan

Runner finished. Started 2026-10-09 23:13 PDT, deadline 06:30, ended 23:39 (26 min).

2 of 3 job(s) ok.

| # | job | result | ran | est | queued at rev | log |
|---|---|---|---|---|---|---|
| 1 | gs18g | FAILED rc=1 | 3 min | 15 min | c2891555 | `/Users/francistownsend-merino/goinfer-logs/night/runs/2026-10-09/gs18g.log` |
| 2 | s7-mac-0909 | ok | 3 min | 20 min | 74830779 +dirty | `/Users/francistownsend-merino/goinfer-logs/night/runs/2026-10-09/s7-mac-0909.log` |
| 3 | s13lite-mac-0909 | ok | 4 min | 20 min | 74830779 +dirty | `/Users/francistownsend-merino/goinfer-logs/night/runs/2026-10-09/s13lite-mac-0909.log` |

## Not ok — last 25 lines of each log

### gs18g — FAILED rc=1

```
loaded vision tower for "gemma-3-4b-it" (256 image tokens/image, soft-token id 262144, encoder f32/metal-resident) from /Users/francistownsend-merino/models/gemma-3-4b-it
  int8 glm_ocr/table.png (7.3s): 'The image shows quarterly unit sales data by region, broken down into North, South, East, West, and Central, and a total'
  int8 gemma3_preprocess_image.png (6.8s): 'The image shows a gradient blend of dark purple and green, transitioning to light purple.'
  int8 qwen25vl_preprocess_image.png (7.1s): 'The image shows a gradient blend of dark purple and bright green, transitioning to a light gray.'
  int8 glm_ocr/formula.png (7.5s): 'The image demonstrates the Gaussian integral and its connection to the normal (or Gaussian) distribution, which is a fun'
metal: KV plan: 2 conversation(s) x 4096 positions (0.9 GB left for the vision tower or drafter)
  decode path: metal-resident (int4)
  KV plan: 2 conversations x 4096 positions (4 asked for; 0.9 GB held back for the vision tower)
loaded vision tower for "gemma-3-4b-it" (256 image tokens/image, soft-token id 262144, encoder int8/metal-resident) from /Users/francistownsend-merino/models/gemma-3-4b-it
  f16b glm_ocr/table.png (7.2s): 'The image shows a table presenting quarterly unit sales by region (North, South, East, West, and Central) for the fiscal'
  f16b gemma3_preprocess_image.png (6.5s): 'The image shows a gradient blend of dark purple and bright green.'
  f16b qwen25vl_preprocess_image.png (6.6s): 'The image shows a gradient blend of dark purple, gray, and light green.'
  f16b glm_ocr/formula.png (7.1s): 'The image demonstrates the Gaussian integral and its connection to the normal distribution (also known as the bell curve'
metal: KV plan: 2 conversation(s) x 4096 positions (1.2 GB left for the vision tower or drafter)
  decode path: metal-resident (int4)
  KV plan: 2 conversations x 4096 positions (4 asked for; 1.3 GB held back for the vision tower)
loaded vision tower for "gemma-3-4b-it" (256 image tokens/image, soft-token id 262144, encoder f32/metal-resident) from /Users/francistownsend-merino/models/gemma-3-4b-it
glm_ocr/table.png: first difference at generated token 3: f16 ' a', int8 ' quarterly'; f16's top-3 ' a' 0.830, ' quarterly' 0.170, ' data' 0.000: NOT a near-tie
gemma3_preprocess_image.png: first difference at generated token 10: f16 ' bright', int8 ' green'; f16's top-3 ' bright' 0.495, ' green' 0.324, ' light' 0.159: near-tie
qwen25vl_preprocess_image.png: first difference at generated token 9: f16 ',', int8 ' and'; f16's top-3 ',' 0.486, ' and' 0.337, '/' 0.145: near-tie
glm_ocr/formula.png: first difference at generated token 12: f16 ' distribution', int8 ' ('; f16's top-3 ' distribution' 0.747, ' (' 0.236, ' probability' 0.015: NOT a near-tie
G-S18g: FAIL
finished: 2026-10-09 23:21:59 PDT rc=1

=== END      2026-10-09 23:21:59 PDT  rc=1
```


## Who picks these up

- gs18g: Claude, S18 on the Mac (owner 2026-10-09: re-queue on the directory) · docs/tasks/task-multimodal-support-2026-10.md · the int8 Metal SigLIP tower against the f16 one on Gemma 3 4B from its directory sidecar; the first night's run used the GGUF, whose tokenizer has no <image_soft_token> (VOID)
- s7-mac-0909: Claude (Mac), S7 third pass on 2026-10-09's tree · docs/tasks/task-multimodal-support-2026-10.md
- s13lite-mac-0909: Claude (Mac), S13-lite third pass · docs/tasks/task-multimodal-support-2026-10.md
