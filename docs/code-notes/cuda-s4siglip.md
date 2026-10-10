# cuda/s4siglip: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `cuda/s4siglip`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## s4siglip

Moved from `cuda/s4siglip/siglip_test.go` (the comment above `s4siglip`) on 2026-10-09.

```text
Package s4siglip reads aikit's CUDA SigLIP tower (gpu/visioncuda) at real size, alone. It is its own package because both aikit's visioncuda and goinfer's
own cuda.VisionEncoder (cuda/vision_register.go) register through the one global vision.RegisterResident hook and the LAST registration wins: inside package
cuda the goinfer tower wins, so a test there measures goinfer's tower, not aikit's (found 2026-10-07; the attached type was *cuda.VisionEncoder).
This package imports nothing from goinfer's cuda, so only visioncuda registers. Information for the aikit defect report, not a gate (G-S4q).
```
