- **`gate quick` writes a run record, and `gate quick -replay FILE` regrades it** (SB12, step 5). `quick_run.json` in
  the run directory holds what quick printed before its verdict, each job's stream or lint output and exit code, the
  hidden failures' log tails, and the run's start and end. The verdict is computed by `gradeQuick`
  (`cmd/gate/grade_quick.go`), and a replay prints exactly what the run printed. No verdict changes.
