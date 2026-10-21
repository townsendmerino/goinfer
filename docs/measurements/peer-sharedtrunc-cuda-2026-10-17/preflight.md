context every arm: cell_ctx() = 4096

### 0.5B
- goinfer `{"max_tokens": 64, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 0.8, "top_k": 40, "top_p": 0.9}`
  cmd `<serve> -model bench=/home/francis/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf -backend cuda -addr 127.0.0.1:8099 -ctx 4096`
- Ollama `{"model": "q05", "options": {"min_p": 0.0, "num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 0.8, "top_k": 40, "top_p": 0.9}, "stream": true}`
- llama.cpp `{"max_tokens": 64, "min_p": 0.0, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 0.8, "top_k": 40, "top_p": 0.9}`
  cmd `llama-server --model /home/francis/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf --port 8098 --host 127.0.0.1 --ctx-size 4096 -ngl 99`
- ollama quant: ('Q4_K_M', '4-bit')
- llamacpp quant: ('Q4_K_M', '4-bit')
- goinfer quant asked: ('int4', '4-bit')  (override table {'M35': 'int4mix', 'phi3-mini': 'q4k'})
- quant_mismatch: ''

### gemma3-1b
- goinfer `{"max_tokens": 64, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 0.8, "top_k": 40, "top_p": 0.9}`
  cmd `<serve> -model bench=/home/francis/models/gemma3-1b-q4_k_m.gguf -backend cuda -addr 127.0.0.1:8099 -ctx 4096`
- Ollama `{"model": "g31b", "options": {"min_p": 0.0, "num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 0.8, "top_k": 40, "top_p": 0.9}, "stream": true}`
- llama.cpp `{"max_tokens": 64, "min_p": 0.0, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 0.8, "top_k": 40, "top_p": 0.9}`
  cmd `llama-server --model /home/francis/models/gemma3-1b-q4_k_m.gguf --port 8098 --host 127.0.0.1 --ctx-size 4096 -ngl 99`
- ollama quant: ('Q4_K_M', '4-bit')
- llamacpp quant: ('Q4_K_M', '4-bit')
- goinfer quant asked: ('int4', '4-bit')  (override table {'M35': 'int4mix', 'phi3-mini': 'q4k'})
- quant_mismatch: ''

### phi3-mini
- goinfer `{"max_tokens": 64, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 0.8, "top_k": 40, "top_p": 0.9}`
  cmd `<serve> -model bench=/home/francis/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf -backend cuda -addr 127.0.0.1:8099 -ctx 4096`
- Ollama `{"model": "p3m-local", "options": {"min_p": 0.0, "num_ctx": 4096, "num_predict": 64, "seed": 1, "temperature": 0.8, "top_k": 40, "top_p": 0.9}, "stream": true}`
- llama.cpp `{"max_tokens": 64, "min_p": 0.0, "model": "bench", "seed": 1, "stream": true, "stream_options": {"include_usage": true}, "temperature": 0.8, "top_k": 40, "top_p": 0.9}`
  cmd `llama-server --model /home/francis/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf --port 8098 --host 127.0.0.1 --ctx-size 4096 -ngl 99`
- ollama quant: ('Q4_K_M', '4-bit')
- llamacpp quant: ('Q4_K_M', '4-bit')
- goinfer quant asked: ('q4k', '4-bit')  (override table {'M35': 'int4mix', 'phi3-mini': 'q4k'})
- quant_mismatch: ''

CHECK OK: temperature, top_p, top_k and seed on every arm; min_p on Ollama and llama.cpp (goinfer has no min_p field)
