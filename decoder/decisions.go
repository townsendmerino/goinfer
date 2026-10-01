package decoder

// DecisionHeadQuant is the weight precision a model with a trained decision head loads at when no quant was chosen
// (`goinfer-chat decide --head`, and `goinfer-serve`'s `head=`); an explicit `--quant` or `quant=` still wins.
//
// The owner decided it on 2026-10-01 from D6b (docs/measurements/decisions-d6b-2026-09/results.md), where Route B on
// JEV-9B was graded against the transformers f32 reference on 150 items. f32 was exact; int8int8 read mean KL 0.009 and
// top-1 agreement 92.7%; int4 read 0.030 and 94.0%; and no arm was resolvably worse calibrated than the reference. The
// pre-registered rule names int8 as the decision models' fallback, and int8int8 fits where f32 (about 36 GB for
// JEV-9B) does not. The capability matrix renders this constant, so the two cannot disagree.
const DecisionHeadQuant = "int8int8"
