- **`chat.Template.ToolGrammar(tools, tokens, stopIDs)` builds the masker for a turn that may call a tool.** It masks
  nothing until the template's call opener (or, for Llama 3, a call object's name key), then holds the call to the offered
  tools' schemas: what `serve` does for `tool_choice: "auto"`, which now calls it. `ParseToolCalls` documents that a call's
  `ID` is the one the model wrote, usually none, and that its content keeps a leading reasoning block. Experimental.
