- **Metal charges the batched step's scratch on every build, and a recurrent model's slot 0 state in its memory guard.** A dense
  build of two or more KV slots used to build the step's scratch uncharged. It now fits it against the memory guard like every other
  build. Serve's default slot count gives up a slot to leave room for it. A slot count asked for with `--kv-sessions` keeps its
  slots, and a miss then turns the batched step off, with a line saying so, rather than refusing the load. `Model.ResidentNeedBytes`
  counts one slot's recurrent state, so Metal's guard and `Plan` price a one-slot recurrent load the same.
