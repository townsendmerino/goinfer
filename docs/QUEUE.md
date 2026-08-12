# Work queue — the shared, claimable list

> **Why this file exists.** The queue used to live in conversation, where only the top of it gets
> restated each turn and everything below silently sinks. Three items aged out that way — the Metal
> consumer window, the out-of-tree consumer audit, the drain fix's CUDA verification — none through
> carelessness. And two boxes pulling from the same unstated queue independently built two
> mechanisms for running the heavy tier, because neither could see the other's progress.
>
> That makes the conversational queue an instance of the class this fortnight has been cataloguing:
> an artifact that exists and is not composed into any decision. A check that cannot fail.
>
> **This file is the queue.** If it is not written here, it is not queued.

## How to use it

- **Claim before starting.** Move the item to `In flight` and put your box and the date on it.
  A claim is what stops the other box duplicating it.
- **Release on finish** — move to `Done` with the commit, or back to `Queued` with what you learned.
- **Never delete an item to tidy up.** Strike it with a reason, so "we decided not to" is
  distinguishable from "it sank".
- **Add the whole item, not a title.** Enough that whoever picks it up does not have to reconstruct
  the context from a transcript they may not have.

Boxes: `linux` (nvidia-rtx2070s, CUDA) · `mac` (Apple Silicon, Metal).

## In flight

_(none — claim here)_

## Queued

_To be seeded. Francis is writing the initial contents; do not reconstruct from a single box's
recent messages, which is the failure mode this file exists to fix._

## Done

_(append with commit sha and date)_
