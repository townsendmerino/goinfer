package serveapp

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

func TestModelFlag_parse(t *testing.T) {
	t.Run("name+path only (back-compat)", func(t *testing.T) {
		var m modelFlag
		if err := m.Set("fast=small.giw"); err != nil {
			t.Fatal(err)
		}
		if m[0].name != "fast" || m[0].path != "small.giw" || m[0].quant != nil {
			t.Fatalf("got %+v", m[0])
		}
	})
	t.Run("bare path", func(t *testing.T) {
		var m modelFlag
		if err := m.Set("plain.gguf"); err != nil {
			t.Fatal(err)
		}
		if m[0].name != "" || m[0].path != "plain.gguf" {
			t.Fatalf("got %+v", m[0])
		}
	})
	t.Run("overrides", func(t *testing.T) {
		var m modelFlag
		if err := m.Set("big=moe.giw,stream,weight-cache=16,quant=int4,kv-quant=i8,embed-int4=false"); err != nil {
			t.Fatal(err)
		}
		s := m[0]
		if s.name != "big" || s.path != "moe.giw" {
			t.Fatalf("identity: %+v", s)
		}
		if s.stream == nil || !*s.stream {
			t.Errorf("stream = %v, want *true", s.stream)
		}
		if s.weightCache == nil || *s.weightCache != 16 {
			t.Errorf("weightCache = %v, want *16", s.weightCache)
		}
		if s.quant == nil || *s.quant != "int4" {
			t.Errorf("quant = %v, want *int4", s.quant)
		}
		if s.kvQuant == nil || *s.kvQuant != "i8" {
			t.Errorf("kvQuant = %v, want *i8", s.kvQuant)
		}
		if s.embedInt4 == nil || *s.embedInt4 {
			t.Errorf("embedInt4 = %v, want *false", s.embedInt4)
		}
	})
	t.Run("errors", func(t *testing.T) {
		for _, v := range []string{"", "p,bogus=1", "p,weight-cache=abc", "p,stream=maybe"} {
			var m modelFlag
			if err := m.Set(v); err == nil {
				t.Errorf("Set(%q): expected error", v)
			}
		}
	})
}

func TestModelSpec_options(t *testing.T) {
	cfg := config{backend: "cpu", quant: "int8int8", kvQuant: "f32", streamWeights: false, weightCacheGB: 0, embedInt4: false}

	// No overrides → inherit defaults.
	base := modelSpec{path: "m.giw"}.options(cfg)
	if base.Quant != "int8int8" || base.StreamWeights || base.EmbedInt4 || base.WeightCacheBytes != 0 {
		t.Fatalf("inherit: %+v", base)
	}

	// Overrides win; an override of "" (f32) is distinct from unset.
	q, gb, st := "", 16.0, true
	s := modelSpec{path: "m.giw", quant: &q, weightCache: &gb, stream: &st}
	o := s.options(cfg)
	if o.Quant != "" {
		t.Errorf("quant override to f32: got %q", o.Quant)
	}
	if o.WeightCacheBytes != int64(16e9) {
		t.Errorf("weightCache: got %d", o.WeightCacheBytes)
	}
	if !o.StreamWeights {
		t.Error("stream override not applied")
	}
	if o.KVQuant != "f32" {
		t.Errorf("kvQuant should inherit default f32, got %q", o.KVQuant)
	}
}

func TestOptionsValidate(t *testing.T) {
	ok := decoder.Options{Backend: "cpu", Quant: "int4", KVPrecision: "i8", KVQuant: "f32"}
	if err := ok.Validate(); err != nil {
		t.Errorf("valid options rejected: %v", err)
	}
	for _, bad := range []decoder.Options{
		{Quant: "int9"},
		{Backend: "tpu"},
		{KVPrecision: "f8"},
		{KVQuant: "int8"}, // common mistake: should be i8
	} {
		if err := bad.Validate(); err == nil {
			t.Errorf("invalid options accepted: %+v", bad)
		}
	}
}
