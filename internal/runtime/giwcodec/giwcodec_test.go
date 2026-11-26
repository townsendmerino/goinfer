package giwcodec

import (
	"errors"
	"testing"
)

func TestLoadSerializedWeights_badMagic(t *testing.T) {
	_, err := LoadSerializedWeights([]byte("NOT_A_GIW_BLOB"), false)
	var se *SerializeError
	if !errors.As(err, &se) {
		t.Fatalf("got err=%v, want *SerializeError", err)
	}
}

func TestCarriesKQuant56(t *testing.T) {
	if !CarriesKQuant56(TargetCPUAmd64) || !CarriesKQuant56(TargetCUDA) {
		t.Fatal("amd64 and cuda should carry Q5_K/Q6_K natively")
	}
	if CarriesKQuant56(TargetMetal) || CarriesKQuant56(TargetCPUArm64) || CarriesKQuant56(TargetNone) {
		t.Fatal("metal, arm64, and none should not carry Q5_K/Q6_K natively")
	}
}
