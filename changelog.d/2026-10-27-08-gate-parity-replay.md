- **`gate parity` writes a run record, and `gate parity -replay FILE` regrades it** (SB12, step 4). The record carries
  the cells' streams and everything else the verdict reads: the checkset after `GATE_RUN`, each gate's ledger class,
  the ledger reconcile text, an `EMIT_MANIFEST` merge's result, and what the sweep printed before the verdict. The
  verdict is computed by `gradeParity` (`cmd/gate/grade_parity.go`), and a replay prints exactly what the sweep printed.
  No verdict changes.
