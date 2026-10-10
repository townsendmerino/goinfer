package decoder

// DecisionHeadQuant is the weight precision a model with a trained decision head loads at when no quant was chosen
// (`goinfer-chat decide --head`, and `goinfer-serve`'s `head=`); an explicit `--quant` or `quant=` still wins. int8int8 is
// the pre-registered fallback for decision models and fits where f32 (about 36 GB for JEV-9B) does not; the evidence is
// docs/measurements/decisions-d6b-2026-09/results.md. The capability matrix renders this constant, so the two cannot
// disagree.
const DecisionHeadQuant = "int8int8"
