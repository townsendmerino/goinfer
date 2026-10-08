//go:build cuda

package cuda

// aikit's CUDA Qwen2.5-VL tower (S4 of docs/tasks/task-multimodal-support-2026-10.md): importing it is the opt-in. gpu/qwencuda plugs into
// vision.RegisterQwenResident, so serve's loadQwenVisionTower can attach it under --backend cuda (G-S4q: correct at real size, 1.6-2.7x the CPU tower).
// aikit's gpu/visioncuda (SigLIP) is deliberately NOT imported: it is wrong at real size (G-S4q), and it would also race goinfer's own SigLIP tower
// (cuda/vision_register.go) for the one global vision.RegisterResident hook, where the last registration wins.
import _ "github.com/townsendmerino/aikit/gpu/qwencuda"
