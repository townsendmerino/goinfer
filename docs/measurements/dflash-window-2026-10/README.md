# The DFlash drafter's sliding, causal layers

Registered 2026-10-26 by nobara-sp2, before any code, on the owner's decision (Francis, 2026-10-26: "2-a": confirm the drafters'
sliding window in z-lab's code, then fix it).

## The finding, confirmed

z-lab's reference inference code (`z-lab/dflash` at `07ebd93db9f472af339b644bb70221ad8428328a`, `dflash/model.py`:
`Qwen3DFlashAttention.__init__` and `_attention_mask`) gives each drafter layer a type from the config's `layer_types`. A
`sliding_attention` layer is:

- **causal** unless the config sets `is_causal`: `self.is_causal = layer_type == "sliding_attention" if is_causal is None`. Over
  the keys `[context; block]`, a block position sees the whole context and the block positions up to and including itself.
- **windowed**: a key is visible only within `sliding_window` positions of the query (both directions when not causal).

A `full_attention` layer is neither (the original DFlash design: non-causal over the context and the whole block). The model
cards say the retrained drafters were "trained with 40k sequence length and sliding-window attention for improved long-context
performance".

goinfer treats every layer as `full_attention`: the CUDA drafter's `attn_block_full` and the CPU trunk (`blockTrunk.layer`)
attend non-causally over the whole context and block, with no window (`cuda/drafter.go`'s attention comment says so). The
drafters on nobara:

| drafter | `layer_types` | window | `is_causal` | differs from the reference |
|---|---|---|---|---|
| `qwen35-9b-dflash` (z-lab, `5fc3b3d4`) | 5 sliding, 1 full | 4,096 | unset → causal | 5 of 6 layers: causal in the block at every context, windowed past 4,096 |
| `qwen36-35b-dflash` (z-lab, `f181eece`) | 5 sliding, 1 full | 4,096 | unset → causal | the same |
| `gemma4-26b-dflash` | 4 sliding, 1 full | 2,048 | unset → causal | 4 of 5 layers |
| `dflash2-qwen38` | 5 sliding | 2,048 | false | windowed only (DFlash 2: refused today) |
| `qwen3-4b-dflash`, `gptoss-20b-dflash`, `laguna-xs2-dflash` | all full or unset | none | | no difference |

Verify is lossless, so the output was never wrong: the drafter drafted from a different attention than it was trained with,
which costs acceptance only. Every speed reading with these drafters (SP0, SP2's G4 and served A/Bs, the 35B) was taken that way.

## The change

- **Load:** `LoadDFlashDrafter` / `OpenDFlashDrafter` read `layer_types`, `sliding_window` and `is_causal`, and give each layer
  its type, causal flag and window (`DrafterGeometry` carries them per layer). A config without `layer_types` keeps today's
  behavior.
- **CPU trunk** (`blockTrunk.layer` and `attend`): the reference mask, per layer.
- **CUDA** (`residentDrafter`'s block attention): the same mask, per layer: causal rows within the block and a window start on
  the context, through the existing `attn_batched` (causal) pipeline or a windowed variant of `attn_block_full`; whichever the
  parity gate passes first, chosen by the code, not by speed.
- **A test hook** turns the masks off (every layer full, today's behavior), for G3's in-process comparison only.

## Gates

- **G1 — parity with the reference (tier 0, then tier 1).**
  - **Tier 0, the mask itself:** a tiny drafter config with a window of 3 and mixed layer types: the CPU trunk's visible-key
    sets equal `_attention_mask`'s, row by row, for a causal sliding layer, a non-causal sliding layer, and a full layer, at
    context lengths under, at and past the window.
  - **Tier 1, the real drafter against z-lab's code:** a golden from `dflash/model.py` at `07ebd93d` (`~/g4venv`, torch CPU,
    f32), on Qwen3.5-9B's drafter: one block (16 positions) over fused context rows of 64, 4,000 and 4,500 positions (under,
    near and past the window), the trunk's output rows before the head. Written gzip-compressed (`testdata/`, the large-golden
    rule).
  - **Bar:** goinfer's CPU trunk at cosine ≥ 0.99999 to the golden on every row (both f32), and the CUDA drafter at ≥ 0.999 (its
    int8 activations, as `TestResidentDrafter_fuseParity` bars it). **Plants:** the window off (fails only the 4,500 case) and
    causality off (fails all three) must each go red.
- **G2 — identity (tier 1):** SP2's `TestSP2_identity` on the 9B with the new masks: tokens and state equal plain decode at w = 4
  and 16. It also records committed tokens a round per prompt, for G3.
- **G3 — acceptance and speed (tier 1/2, by day, the timing lock exclusive), new against today's masks in one process.**
  - **Acceptance:** committed tokens a round at w = 4 on SP0's 12 code and math prompts, short and behind the 3,800-token
    preamble, with the masks on and off (the test hook), on the 9B. Reported per context.
  - **Speed:** the whole-generation in-process A/B (TE5(b)), masks on against masks off, w = 4, short and long, 10 A B B A
    pairs, the void rule as SP2's G4.
  - **What decides:** the change merges on G1 and G2: it makes the drafter match the attention it was trained with. G3 is
    reported, with one stop: if the short context's speed median (on ÷ off) falls under 0.98×, it goes to the owner before
    merging, because faithful masks costing speed would contradict the training.
- **The 35B** is re-read afterwards only if the owner asks (its long cell is ungraded by decision).
- **Stopping rule:** fixed; all cases and pairs.

## Cost, stated before any run

- **The golden:** a minute or two of CPU torch (a 6-layer drafter, 4,500 positions of context).
- **G1, G2:** seconds, then G2's ~15 min on the 9B.
- **G3:** acceptance ~10 min; speed: 20 pairs; the 9B's long pairs took ~4.5 min in SP2's G4, the short ~2: about 65 min.
- **Total about 1.5 h, by day.** A fit check first: the 9B with its drafter at 4,608, as SP2 ran it.

## Readings

(none yet)
