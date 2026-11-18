//go:build cuda

package cuda

import (
	"errors"

	"github.com/townsendmerino/goinfer/decoder"
)

var _ decoder.ResidencyPricer = (*cudaBackend)(nil)

// PriceResident (decoder.ResidencyPricer) is the build's own memory plan over the free VRAM its context plan would read now
// (deriveMemory): one slot's context, the KV slots kept, the KV pools as the device charges them and the margin held. It opens no
// device. A refusal comes back as BuildResident returns it: a pinned context whose KV does not fit is errKVWontFit, anything else a
// decline. A MoE build whose context is planned pays moe_route's local-memory pool before its plan; only paying it measures it, so
// this plan leaves it out and its Note says so.
func (b *cudaBackend) PriceResident(m *decoder.Model) (decoder.ResidentPrice, error) {
	free, ok := decoder.FreeBytesFor("cuda")
	if !ok {
		return decoder.ResidentPrice{Unknown: "cuda: no free-VRAM reading; the build sizes its context at load"}, nil
	}
	p, _, err := b.deriveMemory(m, free, 0)
	if err != nil {
		return decoder.ResidentPrice{}, err
	}
	if p.Decline != "" {
		if errors.Is(p.declineErr, errKVWontFit) {
			return decoder.ResidentPrice{}, p.declineErr
		}
		return decoder.ResidentPrice{}, decoder.DeclineResident("cuda: %s", p.Decline)
	}
	mem := &decoder.ResidentMemory{FreeBytes: free, DenseBytes: m.ResidentDenseWeightBytesFor("cuda"), SlackBytes: m.PlanSlack("cuda"),
		KVBytes: p.KVBytes, MarginBytes: p.Margin, ExtraBytes: p.ExtraBytes, KVSlots: p.KVSlots, KVSlotsAsked: p.KVSlotsReq,
		ConversationCtx: p.CtxCap, KVPrecision: m.KVDeliveredFor("cuda")}
	if p.PoolPaid {
		mem.Note = "moe_route's local-memory pool, which the load pays before its plan, is not priced here, so the load's context can be lower"
	}
	return decoder.ResidentPrice{Ctx: p.SlotCtx, Memory: mem}, nil
}
