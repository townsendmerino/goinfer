- **Changes behavior: on CUDA, Gemma 2 decodes on the capped flash-decode lane by default.** `--flash-decode-softcap`, in chat
  and serve, now defaults to on. A model with an attention softcap therefore takes the flash-decode lane, in its capped variant, once a layer
  attends 2,048 keys or more, instead of the exact split-KV lane. On Gemma 2 2B, decode is 1.69× faster at depth 2,048 (143.2 vs
  84.6 tok/s) and level at depth 128 (0.999). The lane's fidelity gate passed at K = 3,900. Its outputs differ from split-KV's,
  so greedy text can change. It costs no memory, because a capped model loads the lane either way. `--flash-decode-softcap=false`
  keeps the exact lane. A library caller's zero `decoder.Options` still leave it off.
