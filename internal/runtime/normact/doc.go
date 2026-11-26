// Package normact holds the CPU forward's row norms (RMSNorm, LayerNorm, in place or into a destination) and its
// elementwise activations (SiLU, ReLU², GELU erf and tanh). It imports no goinfer package; decoder calls it.
package normact
