- **The record-reading gates decide in pure graders** (SB12, step 2). The deciding functions of `gate release`'s gpu,
  parity, quick-full and CI rows, of `gate gpu-records` and of `gate identity -giw bytes` moved unchanged into
  `cmd/gate/grade_records.go`. `TestGraders_arePure` holds every `grade_*.go` file to no exec, environment, clock or file
  access and no call outside a grader file. No verdict changes; every golden reproduces.
