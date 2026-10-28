- **`gate heavy -stream` refuses (exit 2) and names `gate heavy -replay FILE`** (SB12; Francis's decision of
  2026-10-27). A bare `go test -json` stream has no cell exit codes or durations, so it cannot reproduce a heavy run's
  verdict, and before this change it reported RED with nothing counted. Grade a run's record with `-replay`.
  `gate census -stream` is unchanged.
