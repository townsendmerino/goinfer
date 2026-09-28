# C0 results

## qwen2.5-coder-1.5b-instruct-q4_k_m.gguf (metal), 60 tickets; complete 60; mask-only pass emitted identical ids on 36

| field | kind | instances | value tokens (mean) | free tokens (mean) | free fraction | instances with >= 1 free | gate 0 (>= 80%) |
|---|---|---:|---:|---:|---:|---:|---|
| category | enum | 60 | 3.00 | 1.00 | 0.333 | 1.000 | PASS |
| urgent | boolean | 60 | 1.00 | 1.00 | 1.000 | 1.000 | PASS |
| order_count | integer | 60 | 1.00 | 1.00 | 1.000 | 1.000 | PASS |
| refund_amount | number | 60 | 1.58 | 1.58 | 1.000 | 1.000 | PASS |
| customer_name | string | 60 | 4.92 | 3.92 | 0.797 | 1.000 | PASS |
| summary | string (no gold) | 60 | 19.17 | 18.17 | 0.948 | 1.000 | PASS |

C-gate 1 (cost): readout mean 569.6 us, median 260.5 (p10 242.8, p90 1169.3); mask-only decode token mean 13.56 ms, median 13.85 (p10 11.91, p90 14.80); share of a token: mean 4.20% (bar <= 5%): PASS; median-over-median 1.88%

| field | aggregation | correct | wrong | excluded | AUROC | gate 2 |
|---|---|---:|---:|---:|---:|---|
| category | **decision** (primary) | 48 | 12 | 0 | 0.847 | PASS |
| category | first | 48 | 12 | 0 | 0.845 | secondary |
| category | min | 48 | 12 | 0 | 0.845 | secondary |
| category | geomean | 48 | 12 | 0 | 0.845 | secondary |
| category | product | 48 | 12 | 0 | 0.845 | secondary |
| urgent | **decision** (primary) | 38 | 22 | 0 | 0.727 | PASS |
| urgent | first | 38 | 22 | 0 | 0.727 | secondary |
| urgent | min | 38 | 22 | 0 | 0.727 | secondary |
| urgent | geomean | 38 | 22 | 0 | 0.727 | secondary |
| urgent | product | 38 | 22 | 0 | 0.727 | secondary |
| order_count | first | 43 | 17 | 0 | 0.680 | secondary |
| order_count | **min** (primary) | 43 | 17 | 0 | 0.680 | PASS |
| order_count | geomean | 43 | 17 | 0 | 0.680 | secondary |
| order_count | product | 43 | 17 | 0 | 0.680 | secondary |
| refund_amount | first | 60 | 0 | 0 | n/a | secondary |
| refund_amount | **min** (primary) | 60 | 0 | 0 | n/a | insufficient (needs >= 8 correct and >= 8 wrong) |
| refund_amount | geomean | 60 | 0 | 0 | n/a | secondary |
| refund_amount | product | 60 | 0 | 0 | n/a | secondary |
| customer_name | first | 57 | 3 | 0 | 0.789 | secondary |
| customer_name | min | 57 | 3 | 0 | 0.819 | secondary |
| customer_name | **geomean** (primary) | 57 | 3 | 0 | 0.895 | insufficient (needs >= 8 correct and >= 8 wrong) |
| customer_name | product | 57 | 3 | 0 | 0.842 | secondary |

## qwen2.5-7b-instruct-q4_k_m.int4.metal.giw (metal), 60 tickets; complete 60; mask-only pass emitted identical ids on 24

| field | kind | instances | value tokens (mean) | free tokens (mean) | free fraction | instances with >= 1 free | gate 0 (>= 80%) |
|---|---|---:|---:|---:|---:|---:|---|
| category | enum | 60 | 3.00 | 1.00 | 0.333 | 1.000 | PASS |
| urgent | boolean | 60 | 1.00 | 1.00 | 1.000 | 1.000 | PASS |
| order_count | integer | 60 | 1.00 | 1.00 | 1.000 | 1.000 | PASS |
| refund_amount | number | 60 | 1.52 | 1.52 | 1.000 | 1.000 | PASS |
| customer_name | string | 60 | 4.98 | 3.98 | 0.799 | 1.000 | PASS |
| summary | string (no gold) | 60 | 14.65 | 13.65 | 0.932 | 1.000 | PASS |

C-gate 1 (cost): readout mean 541.7 us, median 265.4 (p10 249.8, p90 1194.4); mask-only decode token mean 37.55 ms, median 36.95 (p10 34.83, p90 40.54); share of a token: mean 1.44% (bar <= 5%): PASS; median-over-median 0.72%

| field | aggregation | correct | wrong | excluded | AUROC | gate 2 |
|---|---|---:|---:|---:|---:|---|
| category | **decision** (primary) | 55 | 5 | 0 | 0.967 | insufficient (needs >= 8 correct and >= 8 wrong) |
| category | first | 55 | 5 | 0 | 0.967 | secondary |
| category | min | 55 | 5 | 0 | 0.967 | secondary |
| category | geomean | 55 | 5 | 0 | 0.967 | secondary |
| category | product | 55 | 5 | 0 | 0.967 | secondary |
| urgent | **decision** (primary) | 58 | 2 | 0 | 0.905 | insufficient (needs >= 8 correct and >= 8 wrong) |
| urgent | first | 58 | 2 | 0 | 0.905 | secondary |
| urgent | min | 58 | 2 | 0 | 0.905 | secondary |
| urgent | geomean | 58 | 2 | 0 | 0.905 | secondary |
| urgent | product | 58 | 2 | 0 | 0.905 | secondary |
| order_count | first | 57 | 3 | 0 | 0.942 | secondary |
| order_count | **min** (primary) | 57 | 3 | 0 | 0.942 | insufficient (needs >= 8 correct and >= 8 wrong) |
| order_count | geomean | 57 | 3 | 0 | 0.942 | secondary |
| order_count | product | 57 | 3 | 0 | 0.942 | secondary |
| refund_amount | first | 60 | 0 | 0 | n/a | secondary |
| refund_amount | **min** (primary) | 60 | 0 | 0 | n/a | insufficient (needs >= 8 correct and >= 8 wrong) |
| refund_amount | geomean | 60 | 0 | 0 | n/a | secondary |
| refund_amount | product | 60 | 0 | 0 | n/a | secondary |
| customer_name | first | 60 | 0 | 0 | n/a | secondary |
| customer_name | min | 60 | 0 | 0 | n/a | secondary |
| customer_name | **geomean** (primary) | 60 | 0 | 0 | n/a | insufficient (needs >= 8 correct and >= 8 wrong) |
| customer_name | product | 60 | 0 | 0 | n/a | secondary |

