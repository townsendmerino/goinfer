//go:build cuda

package cuda

// Qwen2.5-VL's vision tower on CUDA is goinfer's own (qwen25_vision.go, on the tower base with the fused attention), registered through vision.RegisterQwenResident from this package. aikit's gpu/qwencuda is NOT imported
// any more: it carried aikit's unfused attention (7.97 s on the 896x896 image against this tower's), and at serve's defaults on the 8 GB card its scratch allocation failed beside the decoder (S7, 2026-10-07). Its
// allocation-failure fix (v0.1.1) stays in aikit. aikit's gpu/visioncuda (SigLIP) is likewise not imported: it is wrong at real size (G-S4q), and cuda/vision_register.go owns the one global vision.RegisterResident hook.
