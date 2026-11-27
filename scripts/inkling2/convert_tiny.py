# Runs llama.cpp's convert_hf_to_gguf on the tiny model, with a stub vocabulary (the tiny model has no tokenizer).
# Usage: LLAMACPP_DIR=<llama.cpp checkout> python3 scripts/inkling2/convert_tiny.py <tiny model dir> <out.gguf>
# Read at llama.cpp 9fa7a33cbb (conversion/inkling2.py).
import sys, json, os
LL = os.environ["LLAMACPP_DIR"]
sys.path.insert(0, os.path.join(LL, "gguf-py"))
sys.path.insert(0, LL)
d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
cfg["loop_config"] = {"core_start": cfg["loop_layer_index_start"], "core_end": cfg["loop_layer_index_end"],
                      "num_passes": cfg["loop_num_passes"], "gate_rank": cfg["loop_gate_rank"], "is_varloop": False}
cfg["sconv_kernel_size"] = cfg["conv_kernel_size"]
cfg["intermediate_size"] = cfg["moe_intermediate_size"]
json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
import convert_hf_to_gguf as C
import gguf
from conversion import inkling2 as I
def set_vocab(self):
    n = self.hparams["unpadded_vocab_size"]
    w = self.gguf_writer
    w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("inkling2")
    w.add_token_list([f"<t{i}>" for i in range(n)])
    w.add_token_types([1] * n)
    w.add_token_merges(["<t1> <t2>"])
    w.add_eos_token_id(self.hparams["eos_token_id"]); w.add_eot_token_id(self.hparams["eos_token_id"])
    w.add_add_bos_token(False)
I.Inkling2Model.set_vocab = set_vocab
sys.argv = ["convert_hf_to_gguf.py", d, "--outtype", "f32", "--outfile", sys.argv[2]]
C.main()
