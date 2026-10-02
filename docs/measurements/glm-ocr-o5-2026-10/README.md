# GLM-OCR O5: structured extraction, 2026-10-02

Task: [`docs/tasks/task-glm-ocr-2026-10.md`](../../tasks/task-glm-ocr-2026-10.md) item O5. **Everything here is on SYNTHETIC INVOICES
rendered by `scripts/gen_glm_ocr_invoices.py`. They are not real scans**: no skew, noise, compression or paper texture, three fixed
layouts, three font families. What they give is exact ground truth. What they cannot say is how accurate the model is on real
invoices. Reported, not gated: nothing in this directory fails a build.

## Provenance

| | |
|---|---|
| machine | nobara-pc, linux/amd64, RTX 2070 SUPER 8 GB, NVIDIA driver 595.91.07 (the re-anchored driver, `cuda-driver-reanchor-59591`); `~/models` on NVMe (the bench set; nothing read from `/srv/models`) |
| checkpoint | `zai-org/GLM-OCR` revision `2e85a62840ccac27daa451df36c736c4636b8628`, `~/models/glm-ocr` |
| goinfer | commit `1aa61eb3` (+ this directory); aikit v1.52.0 (`GOWORK=off go build ./...` builds) |
| int4 arm | `serve` CUDA build (`cuda/cmd/serve`, `-tags cuda`), `--backend cuda --ctx 8192`, decode `cuda-resident (int4)`, vision tower f32 on the CPU |
| f32 arm | `serve` CPU build, `--backend cpu --quant f32`, vision tower f32 on the CPU |
| sampling | `temperature 0` (greedy; the checkpoint's own generation config is `do_sample: false`), `max_tokens 2500`, no seed needed |
| request | one `image_url` data-URI part, **no text part**, `response_format` `json_schema` = `testdata/glm_ocr/invoice.schema.json` (= `constrain.SchemaFromStruct(Invoice{})` of `examples/invoice`; a test pins the equality), `goinfer_confidence: true` |
| documents | 15 PNGs, 900x1100 to 1050x1250 (0.99-1.31 MP), grayscale, 0.55 MB in all; `testdata/glm_ocr/invoices/` (labels, sha256 manifest, seed 20261002) |
| date, thermal | 2026-10-02 by day; no thermal log. **Seconds in the logs are exploratory, taken while other builds and test suites ran on the box: not quotable** |

## What was asked of the model

The prompt is built from the schema (`constrain.TemplateFromSchema`): the card's instruction `请按下列JSON格式输出图中信息:` and a
template of blanks (`""` for a string, `0` for a number, `false` for a boolean, one example element for the line-item list). The
grammar is compiled from the same schema. The page is never described to the model in words.

## Result 1: schema validity

**15 / 15 replies parsed as JSON (int4); 15 / 15 finished `stop`** (15 / 15 at f32 as well, Result 3b). The grammar guarantees the SHAPE, so this is not a
finding about the model; it is reported because it was checked: every key present, every number a number, no reply truncated.
(A reply cut off by `max_tokens` would be a valid prefix and not parse. None was: the longest was 432 tokens against 2,500.)

## Result 2: per-field exact-match accuracy, int4 on CUDA, 15 documents

Source: `eval_int4-cuda.log`, `results_int4-cuda.json` (every reply, every row scored, every confidence record). Regenerate the
tables with `python3 scripts/o5_eval_invoices.py --report-only results_int4-cuda.json`.

**What was normalised, and nothing else.** Strings (`invoice_number`, `date`, `due_date`, `vendor`, `bill_to`, line item
`description`): runs of whitespace collapsed, ends trimmed; case, punctuation and digits must match exactly, and **dates are compared
as printed**. `currency`: a symbol the model wrote is mapped to its ISO code, then exact (every reply already said `USD`/`EUR`/`GBP`).
Money and quantities: the reply's fields are JSON numbers (the grammar cannot emit a `$` or a thousands separator inside one), compared
as numbers to the cent; the ground truth has its separators and symbols removed the same way. `paid`: exact boolean. Line items are
compared positionally.

| field | correct / n | exact-match |
|---|---|---|
| invoice_number | 15 / 15 | 100.0% |
| date | 11 / 15 | 73.3% |
| due_date | 11 / 15 | 73.3% |
| vendor | 15 / 15 | 100.0% |
| bill_to | 14 / 15 | 93.3% |
| currency | 15 / 15 | 100.0% |
| paid | 10 / 15 | 66.7% |
| subtotal | 15 / 15 | 100.0% |
| tax | 15 / 15 | 100.0% |
| total | 15 / 15 | 100.0% |
| line item count | 15 / 15 | 100.0% |
| line item description | 58 / 58 | 100.0% |
| line item quantity | 58 / 58 | 100.0% |
| line item unit_price | 58 / 58 | 100.0% |
| line item amount | 58 / 58 | 100.0% |
| **all fields** | **383 / 397** | **96.5%** |

Per document (fields correct / fields scored; the document score is correct over the 10 header fields, 1 count and 4 per line item):
inv01 23/23, inv02 34/35, inv03 23/23, inv04 19/19, inv05 34/35, inv06 19/19, inv07 27/27, inv08 24/27, inv09 27/27, inv10 24/27,
inv11 18/19, inv12 19/19, inv13 27/27, inv14 32/35, inv15 33/35. Ten documents are perfect. The full list of wrong fields is the
last table of `eval_int4-cuda.log`.

**All 14 wrong fields are three kinds; no amount, quantity, total or description was wrong:**
- **`paid` (5 of the 14): the model never said `true`.** The five stamped documents (inv02, 05, 08, 11, 14) all came back `false`; the ten
  unstamped ones all `false`, correctly. That is 0 / 5 on stamps. (The stamp is drawn grey, rotated, in a box, beside the totals.) Whether
  that is the model not reading a stamp as "paid" or the template's `false` being copied is not separated by this run.
- **Dates (8: `date` and `due_date` in each of inv08, 10, 14 and 15): the model rewrote the printed date in ISO.** `4 Sep 2026` came back `2026-09-04`; `27 Mar 2026` -> `2026-03-27`. Compared
  as calendar dates, 27 / 30 date fields are right (diagnostic 1 in the log): five of the eight misses are only a format the schema did
  not pin, and three are real errors: inv08's `08/11/2026` (Aug 11, US order) and `09/11/2026` read as `2026-11-08` and `2026-11-09`
  (day and month swapped), and inv14's due date `1 Sep 2026` -> `2026-09-12`.
  A `pattern` would pin the format, but the grammar rejects `pattern` (unsupported keyword), so the way to get the printed form is to ask
  for it in your own prompt text.
- **`bill_to` (1): the model added the address.** inv10 (the compact layout, where the customer's street sits on the next line) came back
  `Pioneer Ridge School District\n9 Foundry Way\nDenver, CO 80202`; the label is the name alone. A reasonable reading of an ambiguous field.

## Result 3: f32 (CPU), a stated SUBSET of 3 documents, and the full pass queued

By day only documents **2, 8 and 10** ran at f32 (`eval_f32-cpu-subset-2-8-10.log`; 79-114 s a document, exploratory). **The subset was
chosen because int4 had errors on them (a paid stamp, a date format, an over-long `bill_to`), so it answers "does f32 recover int4's
misses", NOT "how accurate is f32"**: 83 / 89 fields (93.3%) there, against int4's 82 / 89 (92.1%) on the same three.

| the three documents | int4 | f32 |
|---|---|---|
| inv02 | `paid` wrong | `paid` wrong |
| inv08 | `date`, `due_date`, `paid` wrong | the same three |
| inv10 | `date`, `due_date`, `bill_to` wrong | `date`, `due_date` wrong; **`bill_to` right** |

So at f32 the only difference is the over-long `bill_to`. The stamp and the date-format behaviour are the model's, not int4's.
**The full 15-document f32 pass has since run** (below). Until then no f32 figure on 15 documents existed.

## Result 3b: f32 (CPU) on all 15 documents (night queue, 2026-10-02 12:14-12:36 PDT)

Job `glm-ocr-o5-f32-15-invoices` (`run-f32-15.sh`): the pinned CPU `serve` binary (sha256 in `STAGED_f32-15.txt`, staged from goinfer `1aa61eb3`), `--backend cpu --quant f32 --ctx 8192`,
the same 15 documents, schema, request shape and scorer as the int4 arm. Raw: `results_f32-cpu-15.json`, `eval_f32-cpu-15.log`, `serve_f32-cpu-15.log`. **15 / 15 replies parsed, 15 / 15 finished
`stop`. All fields 378 / 397 = 95.2%, against int4's 383 / 397 = 96.5%.** Reported, not gated: f32 is not more accurate here, and the 5-field difference is formatting.

| field | int4 (CUDA) | f32 (CPU) |
|---|---|---|
| invoice_number, vendor, currency, subtotal, tax, total, line item count | 15 / 15 each | 15 / 15 each |
| line item description, quantity, unit_price, amount | 58 / 58 each | 58 / 58 each |
| bill_to | 14 / 15 | **15 / 15** |
| date | 11 / 15 | 8 / 15 |
| due_date | 11 / 15 | 8 / 15 |
| paid | 10 / 15 | 10 / 15 |
| **all fields** | **383 / 397** | **378 / 397** |

The two arms' replies are **byte-identical on 10 of the 15 documents**. The five that differ (differenced field by field from the two results files):
- **inv01, inv07, inv12: int4 kept the printed date (`10/24/2026`), f32 rewrote both dates in ISO (`2026-10-24`).** The dates are the same calendar dates, so they are right as dates and wrong "as printed", which is the scoring rule
  for the table. **Compared as calendar dates both arms read 27 / 30 date fields correctly** (diagnostic 1 in each log); that is the headline for dates, and the as-printed 8 / 15 against 11 / 15 is the format the schema does not pin.
- **inv10: f32 read `bill_to` as the name alone; int4 added the street address.** This is the whole `bill_to` difference, and the only field f32 recovers.
- **inv14: both wrong, differently.** Printed `1 Sep 2026`; int4 wrote `2026-09-12`, f32 `2026-09-13`.

**`paid` is the same at both precisions: false on all five stamped documents (inv02, 05, 08, 11, 14), right on the ten unstamped.** The stamp misread is the model's, not int4's. Per-field confidence (Result 4) at f32:
the five wrong `paid` values sit at or below 0.8546 and every right value at or above 0.9960, AUROC 1.000, the same ordering as int4 (n = 15, 5 wrong: an indication, not a calibration).

**Seconds** (66 to 113 s a document, 1,291 s for the 15, f32 vision tower on the CPU about 45 s of each) are single runs from the night queue with no load or thermal record: not quotable. They match the daytime subset's 79 to 114 s.

## Result 4: per-field confidence (C1) on a vision request

Before this change a request with images and `goinfer_confidence: true` was refused with a 400 (the route did not write the
result back; the confidence capture itself already ran inside the shared decode driver). The vision chat route now writes it back, in
both the buffered reply and the stream's trailing event, for every vision family. On this schema it reports the **integer** fields
(`quantity`) and the **boolean** (`paid`); strings and numbers are not reported by design, and a Go struct cannot express an enum, so no
enum field was evaluated (the C0 gate cleared enum, boolean and integer).

Does low confidence predict a wrong field on these 15 documents (int4)?

| field | n | wrong | min confidence of a right value | max confidence of a wrong value | AUROC |
|---|---|---|---|---|---|
| `paid` | 15 | 5 | 0.9949 | 0.9547 | **1.000** |
| `quantity` | 58 | 0 | 0.8878 | - | n/a: no wrong value to rank |

For `paid` every wrong answer (the five missed stamps: 0.9547, 0.8222, 0.9191, 0.9523, 0.6853) sits below every right one (0.9949 to
0.9984), so a threshold anywhere in 0.96-0.99 flags exactly the five. **n is 15 with 5 wrong, so this is an indication, not a
calibration, and `calibrated` stays `false`.** For `quantity` the 58 values were all right, so nothing can be said about whether low
confidence there predicts error (its lowest was 0.8878, on a right value). At f32 on the three-document subset the same ordering held
(`paid`: the two wrong at 0.8546 or below, the one right at 0.9976).

## Findings the work turned up

1. **A string-typed numeric column makes the grammar loop on whitespace until `max_tokens`, silently.** With `quantity` typed `string`
   the model wants to write the bare number the column holds; the string grammar forbids it; whitespace is legal at every structural
   boundary, so it is the only legal token left and greedy decoding emits it forever
   (`string_typed_quantity_whitespace_runaway.txt`: the reply stops at `"quantity":` and the rest of the budget is spaces; the first
   run, at `max_tokens` 1,500, was identical). No error: `finish_reason: length`. The cure is to type the amounts as numbers, after which the template shows `0` and the same page extracts
   cleanly. The root cause is constrain's unbounded optional whitespace, which is not changed here; it affects every constrained
   request whose schema disagrees with what the model wants to say, and is worth a bound of its own.
2. **Template convention checked on the model:** numbers as `0`, strings as `""`, a boolean as `false`, one example element for a list;
   the model followed all of it (4-space layout, keys in the template's order). The template's key ORDER is the schema document's order
   (a schema built from a Go struct by `SchemaFromStruct` is alphabetical; `TemplateFromStruct` keeps the struct's field order).
3. **Plain OCR dropped the header that extraction read.** O3 found int4 drops `Invoice No: INV-2026-0417` from `Text Recognition:` output on
   `testdata/glm_ocr/invoice.png`; under the extraction prompt int4 on CUDA and on the CPU both returned `INV-2026-0417`.

## Logs in this directory

| file | what |
|---|---|
| `eval_int4-cuda.log`, `results_int4-cuda.json` | the 15-document int4 CUDA pass |
| `eval_f32-cpu-subset-2-8-10.log`, `results_f32-cpu-subset-2-8-10.json` | the f32 subset |
| `run-f32-15.sh` | the queued full f32 pass (not started) |
| `string_typed_quantity_whitespace_runaway.txt` | finding 1 |
| `chat_demo_cuda_int4.{stdout,stderr}` | the one-line demo, CUDA int4: `goinfer-chat --model ~/models/glm-ocr --image testdata/glm_ocr/invoice.png --schema testdata/glm_ocr/invoice.schema.json` |
| `chat_demo_test_green.log`, `example_test_green.log`, `serve_extraction_test_green.log` | the three real-checkpoint gates (CPU int4), green |
| `mutations_*.log` | each new gate shown red under a deliberate break |
| `test_*.log`, `vet_staticcheck.log` | package test runs (`-v`) and static checks |
