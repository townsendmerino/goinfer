package archparts

// Runner-shape decline reasons. The hardware matrix prints them, so they are short and name no backend.
const (
	RunnerShapeOwnForward = "own forward, not bridged"
	RunnerShapeBringUp    = "bring-up gate (GOINFER_SSM_RESIDENT unset)"
	RunnerShapeNemoMoE    = "Nemotron MoE block"
	RunnerShapeMoE        = "MoE shape"
	RunnerShapeRoPE       = "local and global RoPE tables differ in length"
)
