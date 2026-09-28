# Phase 5 — Design Decisions Log

This document records the clarifying questions raised before implementing Phase 5
(speculative decoding), the options considered for each, and the reasoning behind
the chosen option. It also references the Phase 4 decisions for continuity, since
they directly constrain Phase 5.

The decisions were made "as a CTO" — optimising for **measured performance on the
real target hardware (M5 16 GB)**, **the project's first-class constraint of zero
quality compromise**, and **lowest implementation risk**.

---

## Phase 4 decisions (context — already shipped)

Phase 5 builds directly on these, so they are summarised here:

| Question | Decision | Reason |
|---|---|---|
| Keep resident layers on the Metal GPU (MPS) or in CPU RAM? | Neither — see below | MPS-resident fragmented the Metal allocator (0.041 tok/s). CPU-RAM-resident triggered the macOS memory compressor (0.080 tok/s). Both were ~6–12× slower than the all-streaming baseline. |
| Should partial residency be allowed when the model only partly fits RAM? | **No — full-model-fit guard** | On a 16 GB M5 a 14 GB model leaves no headroom; locking a subset of layers in RAM evicts the OS page cache the *streaming* layers need. `plan_residency()` now returns `resident_count=0` unless the whole model fits the usable budget. |

**Key Phase 4 finding that constrains Phase 5:** on this hardware **RAM is the
binding constraint**. Any design that permanently occupies a large block of RAM
reintroduces macOS memory compression and a severe slowdown. Phase 5 must not add
a large resident object.

---

## Phase 5 — Question 1: Draft strategy

**Question.** Speculative decoding needs a fast token *proposer*. Mistral-7B-Instruct-v0.2
has no official small sibling, and an exact draft must emit token IDs in Mistral's
32 000-entry vocabulary. Which drafting approach should be implemented?

**Options considered.**

| Option | Memory cost | Tokenizer risk | Notes |
|---|---|---|---|
| A. Prompt-lookup (n-gram) decoding | **0 bytes** | **None** | No draft model. Proposes tokens by matching the recent context against earlier n-grams in the prompt + generation. |
| B. Small draft model (e.g. TinyLlama-1.1B) | ~2.2 GB resident | **High** | TinyLlama uses the Llama-2 tokenizer — *not* byte-identical to Mistral's. Mismatched IDs → near-zero acceptance. |
| C. Both, config-selectable | — | — | Build A now, wire a switch for B later. |

**Decision: Option A — Prompt-lookup (n-gram) decoding.**

**Reasoning.**

1. **Memory is the binding constraint (Phase 4).** Phase 4 conclusively measured
   that locking a large object in RAM on a 16 GB M5 triggers the macOS memory
   compressor and a 6–12× slowdown. A 2.2 GB permanently-resident draft model
   reintroduces *exactly* the failure mode Phase 4 just diagnosed and fixed.
   Prompt-lookup has a **zero-byte memory footprint**.
2. **Tokenizer correctness is free.** Prompt-lookup operates on the *target
   model's own emitted token IDs* — there is no second vocabulary, so verification
   is exact by construction. A real draft model with a different tokenizer
   (TinyLlama's Llama-2 tokenizer ≠ Mistral's) would propose IDs that mean
   different things, giving near-zero acceptance — a draft model that is *worse
   than no draft model*.
3. **The disk sweep dwarfs the draft cost.** A speculative step is one ~2 s,
   32-layer SSD sweep. The proposer's cost must be negligible against that. An
   n-gram dictionary lookup is microseconds; a 1.1B-model forward is tens to
   hundreds of ms of pure, non-overlapped overhead.
4. **It is genuinely lossless.** Whatever prompt-lookup proposes, the target model
   verifies greedily — output is bit-identical to plain greedy SWLP. The draft
   only affects *speed*, never *correctness*.
5. **The workload favours it.** SWLP's value proposition is long-context FP16
   inference on consumer hardware — document QA, summarisation, code completion,
   retrieval-augmented chat. These are exactly the repetition-heavy workloads
   where prompt-lookup excels (the model frequently re-emits spans of the context).
6. **Lowest risk, fastest to ship, easiest to test.** No model download, no
   network dependency, no new package, fully deterministic, unit-testable with
   plain tensors.

Option C was rejected as speculative abstraction — CLAUDE.md forbids building
abstractions before a phase requires them. The drafter is a single focused class;
if a model-based drafter is ever justified, that is a separate task.

---

## Phase 5 — Question 2: Verification mode

**Question.** How should the target model verify (accept / reject) draft tokens?

**Options considered.**

| Option | Quality guarantee | Testability |
|---|---|---|
| A. Greedy only | Bit-identical to greedy SWLP output | Exact string equality |
| B. Greedy + sampling | "Same distribution in expectation" | Statistical, multi-seed, flaky |

**Decision: Option A — Greedy verification only.**

**Reasoning.**

1. **The Phase 5 spec is explicit:** "accepted tokens are identical to greedy
   big-model output"; the checklist requires "accepted-token output bit-exact vs.
   greedy SWLP".
2. **Zero quality compromise is SWLP's first-class constraint.** Greedy
   verification yields *bit-identical* output to plain greedy SWLP — the strongest
   possible correctness guarantee, verifiable with a single string-equality check.
3. **Sampling-based speculative decoding** uses modified rejection sampling;
   "lossless" weakens to "same distribution in expectation". That cannot be
   asserted bit-exactly — only via statistical tests over many seeds, which are
   expensive and flaky.
4. **The target config is already pure greedy.** `swlp_mistral_mps.toml` uses
   `temperature = 0.0`, `do_sample = false`, `repetition_penalty = 1.0`. There is
   no sampling behaviour to preserve.
5. **Superset for later.** Sampling support is strictly additive and can be added
   in a future phase if a research need arises — not building it now follows the
   "only build what the phase requires" rule.

To stay bit-exact even if a future config enables a repetition penalty, the
verifier applies the runner's *actual* token-selection function per position
(with the correctly growing prefix), not a bare `argmax`.

---

## Phase 5 — Question 3: KV cache handling

**Question.** Verification processes K draft tokens in one forward; rejected
tokens must be rolled out of the KV cache. The Phase 2 `CompressedDynamicCache`
compresses KV per-layer and is hard to roll back. How should the speculative path
handle the KV cache?

**Options considered.**

| Option | Rollback mechanism | Risk |
|---|---|---|
| A. Plain `DynamicCache`, no compression | `DynamicCache.crop()` — a tested primitive | Low |
| B. Add rollback to `CompressedDynamicCache` | Decompress → crop → recompress every layer, every step | High |

**Decision: Option A — Plain `DynamicCache`, no KV compression on the speculative path.**

**Reasoning.**

1. **KV compression buys almost nothing here.** Phase 2 measured only a **1.10×**
   ratio — zlib is lossless but KV activations are high-entropy. The speculative
   path loses negligible memory by not compressing.
2. **`crop()` is a tested rollback primitive.** `DynamicCache.crop(max_length)`
   is a first-class transformers operation. The compressed cache compresses each
   layer's KV immediately after use (the Phase 2 design); rolling it back means
   decompress → crop → recompress for every layer on every speculative step — both
   complex *and* slow on the hot path.
3. **Correctness risk.** `CompressedDynamicCache` has subtle invariants (e.g.
   `get_seq_length` answers from a recorded length while cold so the attention
   mask builds correctly). Adding crop/rollback to that state machine is a prime
   source of hard-to-find bugs. Speculative decoding is already a non-trivial
   control-flow change; stacking compressed-cache rollback on top multiplies the
   risk surface for no measured benefit.
4. **Separation of concerns.** KV compression (Phase 2) solves "KV too big for
   RAM". Speculative decoding (Phase 5) solves "too many disk sweeps per token".
   They address different bottlenecks; coupling them now violates Single
   Responsibility with no payoff.
5. **They are not both needed for the current target.** For 7B on 16 GB, the KV
   cache for a few-hundred-token generation is well under 1 GB — KV size is not
   the binding constraint. The speculative path can safely use a plain cache.

KV compression and speculative decoding are therefore **mutually exclusive** in
this phase. If a future model genuinely needs both, that is a separate workstream.

---

## Resulting architecture

- `core/speculative.py` — pure logic: `NgramDrafter` (prompt-lookup proposer) and
  `verify_greedy()` (accept/reject + rollback decision). No I/O, no torch model
  calls — unit-testable with plain Python.
- `runner/speculative.py` — `SpeculativeRunner(SWLPRunner)`: reuses the streaming
  scheduler, adapter, and loader; overrides only the decode loop. `backend = "speculative"`.
- One disk sweep per speculative step verifies up to K draft tokens; accepted
  tokens are amortised over that single sweep.
- Output is bit-identical to greedy SWLP — speculation changes throughput only.
