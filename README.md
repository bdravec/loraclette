# Apertus architecture and adapter positions

Apertus is a fully open, multilingual LLM from EPFL, ETH Zurich and CSCS. It is a dense, decoder-only transformer released in two sizes, 8B and 70B.

![Apertus architecture with adapter positions](apertus_architecture_adapters.png)

## Pipeline

1. **Tokenizer**: byte-level BPE with a 131,072-token vocabulary.
2. **Token embeddings**: map each token to a vector (input and output embeddings are not tied).
3. **Decoder blocks × N**: the core of the model (32 blocks in 8B, 80 in 70B).
4. **Final RMSNorm + LM head**: turns the last hidden state into next-token probabilities.
5. **Autoregressive loop**: each generated token is appended to the input.

## Decoder block

Each block is pre-norm and has two sublayers, each wrapped in a residual connection:

- **RMSNorm → grouped-query self-attention**: causal attention with rotary positions (RoPE) and QK-norm; fewer key/value heads than query heads.
- **RMSNorm → feed-forward MLP**: non-gated MLP with the xIELU activation.

No bias terms are used anywhere.

| | Blocks | Hidden dim | MLP dim | Heads (query / KV) |
|---|---|---|---|---|
| Apertus 8B | 32 | 4096 | 21,504 | 32 / 8 |
| Apertus 70B | 80 | 8192 | 43,008 | 64 / 8 |

Context length is 65,536 tokens in v1.0 and 262,144 in v1.5. Version 1.5 also accepts image and audio tokens as input; output stays text only.

## Adapters

An adapter is a small trainable bottleneck (down-projection, nonlinearity, up-projection) added to the frozen model. Only the adapter weights are trained.

Positions in the figure, from most to least common:

1. Beside or after the feed-forward MLP (standard choice)
2. After self-attention, before the residual sum
3. Before the decoder stack, on the embeddings
4. After the decoder stack, before the final RMSNorm
5. On the residual stream, after the sum
6. Between RMSNorm and self-attention

Ranks 1 and 2 follow published adapter comparisons; the order of 3 to 6 is a judgement call and was not benchmarked on Apertus.

## Sources

- [Apertus v1 technical report (arXiv 2509.14233)](https://arxiv.org/abs/2509.14233)
- [Apertus v1.5 model card](https://huggingface.co/swiss-ai/Apertus-v1.5-70B)
- [Code: github.com/swiss-ai](https://github.com/swiss-ai)
