- **Internal packages regrouped under `internal/app/`, `internal/guard/`, `internal/diag/`, and `internal/runtime/`.** As part of the
  Tier 2 repository layout move, root-module internal packages moved from `github.com/townsendmerino/goinfer/internal/<pkg>` to four
  subtrees (`package` clauses unchanged):
  - `internal/{serveapp,chatapp,cliutil,fitcmd,decidecmd,pullcmd,inproc,connect,loadflags}` -> `internal/app/<pkg>`
  - `internal/{swapguard,prezerolint,paritykey,movedpaths,servecheck,memgate}` -> `internal/guard/<pkg>`
  - `internal/{hwcensus,hwreport,kernelprint,sb2census,towerid}` -> `internal/diag/<pkg>`
  - `internal/{giw,mem,kvmgr,f16,mamba2,whisper,scheduler,adapter,hybridattn,program,modelload,prequant,diskfree,sampling}` -> `internal/runtime/<pkg>` (`internal/mem/probe` -> `internal/runtime/mem/probe`)
  - Go enforces `internal/` visibility at the `github.com/townsendmerino/goinfer` module root, so external callers outside the repository could not import these paths; in-tree submodules (`backend/cuda`, `backend/metal`, `backend/webgpu`) updated their imports to the new paths.
