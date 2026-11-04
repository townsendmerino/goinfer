//go:build cuda

package cuda

// Qwen2.5-VL's vision tower on CUDA is goinfer's own (qwen25_vision.go, on the tower base with the fused attention),
// registered through vision.RegisterQwenResident from this package. aikit's gpu/qwencuda is deliberately not imported:
// it carried aikit's unfused attention, and at serve's defaults on the 8 GB card its scratch allocation failed beside
// the decoder. aikit's gpu/visioncuda (SigLIP) is likewise not imported: it is wrong at real size, and
// cuda/vision_register.go owns the one global vision.RegisterResident hook.
