- **A generation that leaves the resident path says why.** `decoder.Generation.ResidentDeclined` names the reason a
  generation ran on the CPU on a model with a resident runner: its prompt passed the resident's context, another generation
  held the resident, or an adapter failed to bind. The first prompt past the resident's context also logs a warning, since
  it is correct but far slower to the first token. Experimental.
