# internal/scheduler: design notes

The package's contract is in its `doc.go`. This file holds the design: why it has this shape, what it does not decide,
and where its evidence lives. Plan: `docs/tasks/task-subsystems-2026-10.md` §8, SB6. Background:
`docs/tasks/task-work-queue-2026-09.md`, J1 and J6.

## Why decisions over plain values

The policy used to live in `internal/serveapp` beside the HTTP code, where the only way to see an ordering was to run
the server. Here it takes numbers and token lists and returns indices and booleans, so the census can replay recorded
traces on a virtual clock, and a future policy (J7's classes and deadlines, RF's batch building) can be tried against
the same traces before anything loads a model. What it cannot hold stays in the server: the session LRU's sessions and
their disk tier (`internal/serveapp/sessions.go:sessionLRU`), the halt state, jobs and batches.

## Admission

`Admission` replaces the plain mutex the server once took, which cannot notice a waiter's client disconnecting, or a
halt, until the waiter gets the lock; by then the dead request has held its place and delayed everyone behind it.
`Enter` drops out the instant its context ends. It is modeled on `golang.org/x/sync/semaphore.Weighted`'s `Acquire`,
specialized to weight 1: the mechanism is reused, not the package, since a dependency is the wrong trade for a primitive
this small (the work-queue doc's rule, no new root-module dependency).

`Release` hands the turn straight to the front waiter without decrementing the held count. That keeps a concurrent
`Enter` from seeing a free turn while others wait. The census records a consequence: lowering `SetCap` takes hold only
as the queue empties, and raising it wakes no waiter. The server sets the cap once, at load.

`Admit` checks for a halt after the grant as well as before admission (the server's `haltGate`), because a halt can
land while a request waits. The halt callback returns whether a halt is on, not only a reason, so a halt with an empty
reason still refuses (`TestAdmit_haltWithNoReasonRefuses`).

## Placement

`BestExtend`'s floor keeps conversations that open alike from evicting each other's turns: a candidate must match the
prompt for longer than it matches every other resident session. With a single resident session the floor knows no
preamble, so `PickSession` adds the room rule (with a spare slot, a fresh session beats overwriting most of another's
history) and its exception for a conversation rolling back its own history (`KeepsFirst`). The census records what the
room rule allows: two conversations with a long shared preamble and little history of their own trade one session,
because each keeps more of the other's session than it overwrites.

`PickDonor`'s `>=` makes the least recent of equal donors win. That is the session a fresh slot would evict, so the
server takes it in place, cut back to the shared rows, and saves the copy.

## Not decided here

- **Reordering on the prompt (J6).** Preferring the waiter that shares the longest prefix with a warm session was
  measured and killed (`docs/measurements/j6-prefix-scheduling-2026-09-15.md`). Admission is strict FIFO, and
  `Record.PromptIDs` is carried so a later policy has it.
- **Classes, priorities, deadlines (J7).** Not built. The census traces carry each request's session, class and
  deadline next to the FIFO order, so a policy that uses them shows its whole effect as a golden diff, which needs
  Francis's word.
- **The resident GPU slots.** `internal/kvmgr`'s `PickSlotFor` places a request on a GPU KV slot; it is the same kind of
  decision over plain values, already outside the HTTP package.
