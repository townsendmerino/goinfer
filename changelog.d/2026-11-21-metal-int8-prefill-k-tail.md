- **Fixed: Metal's int8 prefill read past the end of a row when K was not a multiple of 32.** The native-int8 batched
  prefill GEMMs summed a few weights and activations from the next row, and at the last row read leftover scratch, so the
  result could depend on what ran before. It took `-quant int8int8` on Metal, a batched prefill and such a projection
  width; no released checkpoint we checked has one. A K that is a multiple of 32 gives the same bits as before (#1206).
