package audio

import (
	"testing"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// TestWhisperWalkBlock_refusesMisroutedOps: the encoder's walker refuses each misrouted op with an error. The layer's norm
// weights are nil, so a norm that ran would panic.
func TestWhisperWalkBlock_refusesMisroutedOps(t *testing.T) {
	const hidden, inter = 8, 32
	e := &WhisperEncoder{Cfg: WhisperEncoderConfig{DModel: hidden, FFN: inter, Heads: 2}}
	sq := make([]float32, hidden*hidden)
	ly := &whisperLayer{qW: sq, kW: sq, vW: sq, oW: sq, fc1W: make([]float32, inter*hidden), fc2W: make([]float32, hidden*inter)}
	for _, c := range vision.MisroutedTowerOpsForTest() {
		err := func() (err error) {
			defer func() {
				if recover() != nil {
					t.Errorf("%s: ran instead of refused", c.Name)
				}
			}()
			return e.walkBlock(vision.TowerBlock{c.Op}, ly, make([]float32, 4*hidden), 4, 4)
		}()
		if err == nil {
			t.Errorf("%s: accepted", c.Name)
		}
	}
}
