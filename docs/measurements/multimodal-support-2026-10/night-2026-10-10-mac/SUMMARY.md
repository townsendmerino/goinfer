# Night run 2026-10-10 — macbookpro.lan

Runner finished. Started 2026-10-10 12:29 PDT, deadline 06:30, ended 13:38 (1 h 09 min).

4 of 5 job(s) ok.

| # | job | result | ran | est | queued at rev | log |
|---|---|---|---|---|---|---|
| 1 | gs18g2 | FAILED rc=3 | 32 min | 1 h 00 min | 5839762a | `/Users/francistownsend-merino/goinfer-logs/night/runs/2026-10-10/gs18g2.log` |
| 2 | s7-mac-1010 | ok | 3 min | 20 min | fe2c9c53 | `/Users/francistownsend-merino/goinfer-logs/night/runs/2026-10-10/s7-mac-1010.log` |
| 3 | s13lite-mac-1010 | ok | 4 min | 20 min | fe2c9c53 | `/Users/francistownsend-merino/goinfer-logs/night/runs/2026-10-10/s13lite-mac-1010.log` |
| 4 | metal-q4k-gq-2 | ok | 1 min | 45 min | eee75d83 +dirty | `/Users/francistownsend-merino/goinfer-logs/night/runs/2026-10-10/metal-q4k-gq-2.log` |
| 5 | giw-gt-mac | ok | 6 min | 20 min | 99c1ea24 | `/Users/francistownsend-merino/goinfer-logs/night/runs/2026-10-10/giw-gt-mac.log` |

## Not ok — last 25 lines of each log

### gs18g2 — FAILED rc=3

```
qwen25vl_preprocess_image_resize.png 'What does this image'    13   0.0612   0.4991  +0.4379    12/ 10          0.097-0.329
qwen25vl_preprocess_image_resize.png 'Describe this image '    23   0.0614   0.1760  +0.1146    21/ 20          0.068-0.082
qwen35vl_preprocess_image.png      'What does this image'    17   0.0865   0.0700  -0.0165    14/ 16          0.059-0.144
qwen35vl_preprocess_image.png      'Describe this image '    28   0.0667   0.0622  -0.0045    27/ 27          0.063-0.080
glm_ocr/invoice.png                'What does this image'    25   0.2436   0.1061  -0.1375    23/ 23          0.180-0.326
glm_ocr/invoice.png                'Describe this image '    32   0.0620   0.0970  +0.0350    29/ 30          0.059-0.128
glm_ocr/invoices/inv01.png         'What does this image'    32   0.0265   0.0183  -0.0081    32/ 32          0.027-0.048
glm_ocr/invoices/inv01.png         'Describe this image '    32   0.0335   0.0439  +0.0105    31/ 31          0.030-0.104
glm_ocr/invoices/inv02.png         'What does this image'    32   0.0899   0.1777  +0.0877    29/ 30          0.055-0.071
glm_ocr/invoices/inv02.png         'Describe this image '    32   0.1023   0.1286  +0.0263    31/ 30          0.072-0.165
glm_ocr/invoices/inv03.png         'What does this image'    32   0.1139   0.0882  -0.0257    30/ 30          0.098-0.153
glm_ocr/invoices/inv03.png         'Describe this image '    32   0.0516   0.2329  +0.1813    31/ 30          0.066-0.193
glm_ocr/invoices/inv04.png         'What does this image'    32   0.1141   0.0615  -0.0526    31/ 31          0.049-0.084
glm_ocr/invoices/inv04.png         'Describe this image '    32   0.1320   0.0459  -0.0861    29/ 30          0.058-0.130
glm_ocr/invoices/inv05.png         'What does this image'    31   0.0758   0.0389  -0.0369    29/ 30          0.028-0.084
glm_ocr/invoices/inv05.png         'Describe this image '    32   0.0554   0.1102  +0.0548    31/ 32          0.043-0.182

mean KL to the CPU tower: f16 0.0827, int8 0.1417 nats per position
D (int8 minus f16)   +0.0590 [+0.0041, +0.1243]   bands on the upper bound: PASS <= 0.05, PARKED <= 0.2 -> PARKED
A (f16 minus int8)   +0.61 points [-1.15, +2.62]   bands on the upper bound: PASS <= 5.0, PARKED <= 15.0 -> PASS

G-S18g2: PARKED
finished: 2026-10-10 13:03:52 PDT rc=3

=== END      2026-10-10 13:03:52 PDT  rc=3
```


## Who picks these up

- gs18g2: Claude (Mac), S18 G-S18g2: the int8 Metal tower against f16, graded · docs/tasks/task-multimodal-support-2026-10.md
- s7-mac-1010: Claude (Mac), S7 fourth pass on 2026-10-10's tree (Gemma 3's image prefill) · docs/tasks/task-multimodal-support-2026-10.md
- s13lite-mac-1010: Claude (Mac), S13-lite fourth pass · docs/tasks/task-multimodal-support-2026-10.md
- metal-q4k-gq-2: Claude (Mac), metal q4k G-Q2/G-Q3, the fit guard bypassed on the owner's word (2026-10-10), swap watch on · docs/tasks/task-metal-q4k-2026-10.md
- giw-gt-mac: Claude (Mac), giw v16 transcode: G-T1 real cells, G-T2, G-T2c, G-T3 · docs/tasks/task-giw-v16-transcode-2026-10.md
