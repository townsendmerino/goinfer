# Review brief — check every classification, both directions

You are a reviewer who has NOT seen how these classifications were produced. Your job is to check them, not to defend
them. Read the rubric first: /tmp/claude-0/-home-claude/257ad878-dfb0-56e4-943a-97af446d8742/scratchpad/ac/rubric.md

Apply the rubric's R test STRICTLY: R needs (1) an option value, a model family, or a kind of state that EXISTS,
(2) a guard / predicate / dispatch / serializer / loader branch / lifecycle path in PRODUCT code that never accounts
for it, and (3) a fix that is a REGISTRATION — adding it to a list, predicate or switch, or giving it a named decline.
If the fix is honouring a return value, fixing an index or a comparison, moving code, or correcting arithmetic, the
finding is NOT R (it is usually N, or O). Apply B equally strictly: a specific limit that exists in product code and a
reachable input that exceeds it unchecked.

Check BOTH directions with equal care: an R or B that should be N, G or O, AND an N, G or O that should be R or B.
Nobody wants the totals to move in either direction.

For each line of your input file, open the audit entry (search the audit file for the ID; read the full entry, not
only its title) and decide AGREE or CHANGE.

Write JSONL to your output path, one line per input line, same order:
{"id": "C-05", "verdict": "AGREE" | "CHANGE", "to": "<class if CHANGE, else null>", "why": "<one line, <= 30 words>"}

Then reply with ONLY: the output path, the line count, and the number of CHANGE verdicts by from->to.
Do not open any other passA_/passB_/review_out_ files.
