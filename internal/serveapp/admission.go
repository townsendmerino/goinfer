package serveapp

import (
	"context"

	"github.com/townsendmerino/goinfer/internal/scheduler"
)

// admissionRecord is a waiter's scheduler.Record: its prompt token ids, and the id position() names it by (a job's id;
// "" for requests nothing will ask about).
type admissionRecord struct {
	promptIDs []int
	id        string
}

// admission is the model's turn granter, scheduler.Admission (FIFO, context-aware, -max-concurrent wide), with the
// lowercase methods this package calls. The zero value is ready to use and admits one.
type admission struct {
	scheduler.Admission
}

func (a *admission) setCap(n int) { a.SetCap(n) }

func (a *admission) load() int { return a.Load() }

func (a *admission) enter(ctx context.Context, rec admissionRecord) bool {
	return a.Enter(ctx, scheduler.Record{PromptIDs: rec.promptIDs, ID: rec.id})
}

func (a *admission) position(id string) (place, waiting int, running, ok bool) { return a.Position(id) }

func (a *admission) release() { a.Release() }
