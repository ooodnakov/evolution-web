**Bottom line**

Replace the flat mean-pool MLP with a **hierarchical entity-token Transformer** (“HET”): trait rows → parent entity token → masked self-attention over entity tokens → attention-pooled context → **pointer/candidate scorer over the current prefix choices**. Keep the existing autoregressive prefix and dynamic `Discrete(len(choices))`; do not add a fixed action head.

- **Next milestone:** HET-lite, `d=128`, 2 layers, 4 heads, FFN 256, ~0.4–0.7M params. [INFERENCE]
- **Later scale-up:** HET-full, `d=192–256`, 6–8 layers, per-trait tokens + relation bias/GNN edges, public-history/belief tokens, league/PSRO. Perceiver only if token count grows past ~1k.

---

## 1. What this environment actually looks like

Measured from the repo and from rollouts (survival policy; small sample, so treat as sizing evidence, not a benchmark):

| config | sampled | decisions | choices mean / p99 / max | prefix mean / max | animals mean / max | traits max | max traits/animal | plants max | hand max | obs bytes mean / p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base 2p | 2 eps | 1,189 | 1.86 / 6 / 11 | 0.98 / 3 | 1.9 / 2 | 30 | 21 | 0 | 15 | 5.4 KB / 10.3 KB |
| all 2p | 2 eps* | 3,473 | 4.64 / 37 / **126** | 1.16 / 4 | 7.0 / 36 | 60 | 37 | 8 | **128** | 13.4 KB / 22.0 KB |
| all 8p | 1 ep | 2,168 | 3.61 / 17 / 26 | 1.15 / 4 | 14.1 / 28 | 65 | 16 | 16 | 19 | 21.6 KB / 28.7 KB |

\*One all-2p episode hit `ENGINE_STALLED: empty action domain for DEFENSE player_1`; see risks.

Implications:

- **Prefix depth is tiny.** Mean prefix length ~1.0–1.16 → ~2.0–2.2 prefix steps per compound action; max observed 4. Autoregressive prefix overhead is ~2×, not combinatorial.
- **Candidate sets are small but not negligible.** Max 126 choices (all-2p); 26 at 8p; 11 base-2p. A pointer scorer over ≤128 candidates is cheap.
- **Entity population is modest.** Max 36 animals, 65 traits, 16 plants, 128 own hand cards. Entity-token count is ~20 (base-2p), ~73 (8p), ~176 without flat traits / ~236 with flat traits (2p all-addons outlier). Attention is feasible; no Perceiver needed yet.
- Observation size 5–22 KB per decision. JSON → tensor path, not GPU FLOPs, is the practical bottleneck.

Repo facts:

- Current model: `Embedding(16)`, `HIDDEN_DIM=64`, `RowMlp` perceptrons, context 320, choice rows 256, candidate input 576, one value head. Checkpoint param count: **143,234** [MEASURED from `rl/runs/smoke/checkpoint.pt`].
- Encoder mean-pools own animals, all opponents’ animals, and plants. It keeps per-animal vectors for choice gathering, but no animal-animal or trait-trait interaction. See `rl/python/evolution_rl/train_smoke.py:238-258, 461-546`.
- Candidate scorer is already pointer-style but flat MLP: `score(context, choice, source, target, entity)`. See `:548-570`.
- Single `type_embed` is shared across trait/card/plant/command namespaces, although catalog IDs are namespace-local (`rl/node/catalog.js:71-99`). This aliases e.g. trait type id 1 and card type id 1. [INFERENCE from code]
- Hand-card choices store `type_id` and `entity_slot=hand index`, but leave `entity_kind=NONE` (`rl/node/actions.js:117-124`). The current scorer therefore does **not** gather the hand card’s `trait1_type/trait2_type`; it only sees card type and index. [INFERENCE from code]
- Slot order is meaningful and never sorted (`rl/node/slots.js:1-25`); architecture must preserve slot ↔ token alignment.
- Opponent hidden traits are omitted; own hidden traits are present; opponent transient values are masked (`rl/node/observation.js:155-190`). Do not add “unknown” tokens per hidden trait—that would leak hidden counts.

---

## 2. Ranked architectures

### Rank 1 — Hierarchical entity-token Transformer (HET)

**Shape.** Token matrix `X ∈ R^{N×F}`, `N` ≈ 20–236 (use padding only inside batch). Token types: global, player, animal, plant, shell, hand card. Traits are encoded per row (`F_trait≈76` current feature layout) and aggregated into their parent entity by masked mean or one local attention head. Then:

```
H^0 = X W_in + token_type_emb + owner/seat_emb + slot/position_emb   # [N, d]
H^{l+1} = FFN(LN(H^l + MHA(LN(H^l), mask)))                           # L layers
c = [H_global; PMA(H_entities)]                                       # [2d] or [d]
```

Choices are separate: `Z ∈ R^{K_t×d}` for `K_t ≤ 126`; `logits = MLP([c; Z; gathered entity/trait embeddings])` or bilinear `c^T W Z`. Prefix conditioned by a tiny GRU/mean over previous choice embeddings.

**Params.** HET-lite (`d=128`, `L=2`, FFN 256): ~0.4–0.7M. HET-full (`d=192`, `L=6`): ~2.5–4M; `d=256`, `L=8`: ~6–8M. [INFERENCE, analytic]

**Sample efficiency.** Best asymptotic bias for this game: self-attention reasons over relations between animals, plants, traits, and players; relational architectures improved sample complexity and generalization in Box-World/SC2 in [Relational Deep RL](https://arxiv.org/abs/1806.01830). With only 32 episodes, do not expect measurable win-rate gain; use it to remove the representational ceiling.

**Integration.** Python-only: new tokenizer/collator, `spaces.py` unchanged, Node schema unchanged. Need a `(entity_kind, entity_slot, trait_slot) → token index` map; route `HAND_CARD` by `field_kind` because `entity_kind=NONE`. Use separate embeddings per catalog namespace (`trait_type`, `card_type`, `plant_type`, `command_type`). Reuse current `CandidatePolicy` interface.

**Failure modes.** Masked-attention leakage of padding/opponent-hidden tokens; slot order lost if entities are sorted; overfit at 32 episodes; O(N²) attention is fine at N≤250 but wasteful if batch/token counts explode; a flat transformer over all traits without parent embeddings can blur two-level nesting.

### Rank 2 — Hierarchical DeepSets + relation features + pointer scorer

**Shape.** `trait → MLP(76→d)`, aggregate into animal `MLP(5+flags+d→d)`, plant similarly; group means: own animals, each opponent separately, plants, hand, shells; context `[global, own, opp_1..opp_7, plants, hand, shells, context]`. Candidate scorer `[5d+4d]→d→1`. Params ~0.3–0.5M. [INFERENCE]

**Sample efficiency.** Medium. Cheapest strong baseline; good ablation for “does attention help?”. It preserves per-animal trait binding better than the current global mean, but still cannot model animal-animal or trait-trait interactions.

**Integration.** Lowest: modify `ObservationEncoder`; keep `CandidatePolicy`; no schema change. This is the natural fallback if HET overfits the smoke.

**Failure modes.** Mean pooling aliases “which trait on which animal”; all opponents still collapse unless you keep per-opponent groups; no link resolution; ceiling below attention/GNN.

### Rank 3 — Relational GNN / message passing

**Shape.** Nodes = animals + plants + players + shells, `V≤~50`; edges from ownership, position adjacency, trait links (`linked_entity_kind`, `linked_entity_slot`, `link_source`, `covers`); edge features = trait type. `H∈R^{V×d}`, 3 rounds of `m_ij = MLP([h_i,h_j,e_ij])`, `h_i' = MLP([h_i, Σ_j m_ij])`. Params ~0.4–0.8M. [INFERENCE] Background: [Graph Networks](https://arxiv.org/abs/1806.01261), [Relation Networks](https://arxiv.org/abs/1706.01427).

**Sample efficiency.** High **if** explicit links are the bottleneck; otherwise similar to DeepSets with more tuning. Best inductive bias for parasite/mimicry/tail-loss/linked-trait relations.

**Integration.** Highest: Python graph builder from JSON, scatter/gather batching, edge masks. Many rule interactions are implicit (feeding competition, attack resolution), so the graph is incomplete; attention is more general.

**Failure modes.** Missing/incorrect edges; scatter nondeterminism/slowness; overfitting edge types; batching variable sparse graphs is more code than padded attention.

### Rank 4 — Perceiver IO latent bottleneck

**Shape.** Latents `M=32–64 × d=128`; cross-attend `N→M`, self-attend latents, output latents; choice scorer conditions on latents. Params ~0.6–1.2M. [Perceiver IO](https://arxiv.org/abs/2107.14795) scales linearly with input size and was evaluated on StarCraft II.

**Sample efficiency.** Medium-low at this scale: latent bottleneck loses fine entity detail and needs more data. Not justified at N≤250; reconsider if per-trait + public-history tokens push N toward 1k+.

**Integration.** Medium: cross-attention module, masks, latent array; Node/schema unchanged.

**Failure modes.** Bottleneck underfits rare traits; harder optimization; extra hyperparameters; no throughput benefit at current N.

### Rank 5 — Current flat mean-pool MLP (baseline only)

**Shape.** context 320, choice 256, candidate 576→64→64→1; value 320→64→64→1; **143,234 params** [MEASURED]. Keep as the control.

**Failure modes.** All opponents mean-pooled into one vector, so 8p opponent identity is lost; single type embedding aliases namespaces; hand-card scorer lacks card trait sides; no pairwise relations; low ceiling. Still useful as an ablation and smoke baseline.

---

## 3. Action modeling

The current design is already the right family: **autoregressive prefix decoding with candidate scoring**. Each decision is one field; the compound action probability is the chain rule:

\[
\pi(a_1,\dots,a_k\mid s)=\prod_{t=1}^{k}\pi(a_t\mid s,a_{<t}),\qquad
\mathcal L_{\text{policy}}=-\sum_t \log\pi(a_t\mid s,a_{<t})\,\hat A
\]

Measured mean prefix ~2.0–2.2 steps, max 4; candidate max 126. No Cartesian product is ever needed: Node returns only the current-level `choices`, Python exposes `Discrete(max(1,K_t))` per decision.

**Candidate scoring = pointer network.** Score each current choice with a query from the context/prefix state and keys from choice features + gathered entity/trait embeddings:
\[
\ell_i = w_2^\top \tanh(W_1[c;x_i;e_i]+b_1)+b_2
\]
then masked softmax. This is exactly the variable-output-dictionary setting of [Pointer Networks](https://arxiv.org/abs/1506.03134). AlphaStar is the closest published analogue: entity encoders plus an autoregressive action head selecting units/targets pointer-style ([Nature](https://www.nature.com/articles/s41586-019-1724-z); architecture summary in [Rethinking of AlphaStar](https://arxiv.org/abs/2108.03452)).

**Consistency details.**

- Store per-step `(context, choices, mask, chosen)`; recompute logits under the same encoder state.
- Mask padded and invalid choices with `-inf`; entropy must ignore padding.
- For PPO, treating each prefix step as a timestep with the same terminal return (current approach) and clipping per step is a heuristic; exact compound-action importance ratio is the product of per-step ratios. Choose one and document it. [INFERENCE]
- Gamma=1 across prefix steps matches terminal-only reward; no length discount.
- Never cache a candidate set beyond the current state/prefix; choices change after each field.

**Alternatives.**

- **Fixed factored heads** (type head + per-type field heads): workable, but every field set (hand cards, animals, positions, link endpoints) is still a pointer; more code, no clear gain.
- **Branching dueling Q-networks** ([BDQ](https://arxiv.org/abs/1711.08946)): assumes fixed action dimensions; our prefix depth/fields vary → poor fit.
- **SAINT-style sub-action attention** ([SAINT](https://arxiv.org/abs/2505.12109)): useful if sub-action dependencies become strong; our prefix is ordered and short, so candidate scoring is simpler.
- **Large-discrete-action-space embeddings** ([Dulac-Arnold](https://arxiv.org/abs/1512.07679)): unnecessary; explicit candidate sets ≤128 and masks must stay exact.

---

## 4. Hidden information

The observation is **not Markov**: opponent hands, hidden traits, and deck order are absent; two different hidden states can produce the same visible observation. Current schema exposes only the current visible state, not public action history (`rl/schema.json:47-133`). Therefore the current policy is a memoryless approximation, not a full information-state policy. [INFERENCE from schema]

A policy conditioned on own information + public history is a valid behavioral strategy and is enough to train by self-play, but it does **not** guarantee Nash/exploitability in 2–8-player general-sum games. Evidence:

- NFSP approached Nash in Leduc but is empirical, not a general guarantee: [NFSP](https://arxiv.org/abs/1603.01121).
- ReBeL provably converges in two-player zero-sum with search/public belief state: [ReBeL](https://arxiv.org/abs/2007.13544); Deep CFR approximates CFR without abstraction: [Deep CFR](https://arxiv.org/abs/1811.00164); DeepStack uses belief/search: [DeepStack](https://arxiv.org/abs/1701.01724).
- DeepNash is model-free, no search, no explicit belief model, and uses R-NaD to approach Nash in two-player zero-sum Stratego: [DeepNash](https://arxiv.org/abs/2206.15378).
- Suphx handles four-player imperfect-information Mahjong with oracle guiding and runtime adaptation: [Suphx](https://arxiv.org/abs/2003.13590).
- Student of Games unifies search/learning for perfect and imperfect information: [Student of Games](https://arxiv.org/abs/2112.03178).
- Multiplayer general-sum self-play needs population/league methods: [PSRO](https://arxiv.org/abs/1711.00832), AlphaStar league.

**Recommendation for this project.**

- Milestone: own-view policy + stochastic self-play + value baseline. This is honest and sufficient for a learning smoke; do not claim information-state optimality.
- Scale-up: add bounded public history tokens (`K≈32–64` events: actor, command type, source/target slots, revealed trait type, food delta) and an auxiliary belief head over opponent hand composition. Train belief with privileged labels; never expose them to the policy. This is the “asymmetric actor-critic” pattern ([Pinto et al.](https://arxiv.org/abs/1710.06542)) and is related to Suphx’s oracle guiding.
- Full CFR/ReBeL search is not a first-milestone architecture: it needs a public belief state/search tree, and 2–8-player general-sum CFR is impractical. Use game-theoretic methods only for 2p zero-sum diagnostics/exploitability.

---

## 5. Efficiency on one RTX 4080 SUPER

- Current 143k-param MLP; HET-lite ~0.4–0.7M; HET-full ~3–7M. All fit 16 GB trivially. [MEASURED/INFERENCE]
- Back-of-envelope: batch 128–256 decisions, `N≈250`, `d=128`, 2 layers → a few GFLOPs per batch; sub-millisecond to low-millisecond on a 4080. The environment/JSON path dominates. [INFERENCE]
- Measured environment cost: base-2p ~0.7 s/episode (~600 decisions → ~1.2 ms/decision); all-8p ~90 s/episode (~2,168 decisions → ~41 ms/decision) (`rl/runs/smoke/report.json`, rollout samples). GPU will idle unless you batch many envs.
- **Vectorize the observation → tensor path.** Current code creates many small `torch.tensor` objects per trait/entity/choice. Build NumPy arrays once per decision: `token_type[N]`, `features[N,F]`, `parent[N]`, `slot[N]`, `mask[N]`; collate to `[B,N,F]` with padding; one `torch.from_numpy`; one masked forward. This is the main CPU win. [INFERENCE]
- **Batch across 8–16 env subprocesses** (16 cores); use a queue to feed the GPU. Async actor/learner patterns: [IMPALA/V-trace](https://arxiv.org/abs/1802.01561), [Sample Factory](https://arxiv.org/abs/2006.11751), [Podracer](https://arxiv.org/abs/2109.06780).
- **Cache within a decision.** Entity tokens are unchanged across prefix steps; cache them and recompute only the prefix/choice head. If cross-prefix caching is needed, add a `state_token`/`compound_id` to the observation; current decision IDs do not identify the same compound action. [INFERENCE]
- Use pre-LN and LayerNorm, not BatchNorm (variable batch/entity counts). Pad only inside the batch; mask logits with `-inf`; no entity/action caps.
- Candidate scorer: vectorize over `K`; one gather for entity/trait embeddings; do not loop over choices in Python.

---

## 6. Concrete recommended architecture for the next milestone

**HET-lite** — replace `ObservationEncoder` + `CandidatePolicy` encoder path; keep the same training/action interface.

```
Tokens:
  global       1 × (phase, turn, round, food, deck_count, active_players,
                    addons[7], half_deck, seat/current/round rel.)
  player       ≤8 × (seat_emb, playing, ended, acted, hand_count, animal_count,
                     relative-to-self)
  animal       ≤36 × (owner/seat_emb, position_emb, food, visible_food_size,
                      visible_fat_size, flags[8], trait_aggregate[d])
  plant        ≤8 × (type_emb, food, covers, cover_slots, fruit, trait_aggregate[d])
  shell        ≤1 × (type_emb, value, slot)
  hand card    ≤128 × (card_type_emb, trait1_type_emb, trait2_type_emb, slot)
  choices      K_t ≤126, encoded separately (not appended to entity tokens)

Trait row encoder: MLP(76 → d) with its own type embedding; aggregate into parent.
Entity token encoder: type-specific MLP → d; add token-type + owner/seat + slot/position embeddings.
Transformer: L=2, d=128, heads=4, FFN=256, dropout=0.1, pre-LN, masked.
Pool: c = [h_global; PMA_1seed(H_entities)] → context MLP.
Choice scorer: MLP([c; choice_enc; gathered entity/trait]) → logit; masked softmax.
Value head: MLP(c) → scalar.
```

Params ~0.4–0.7M. [INFERENCE] Keep separate embeddings per catalog namespace. Route `HAND_CARD` by `field_kind` to the hand token; route `TRAIT` to the `(entity, trait_slot)` row. Preserve slot order; never sort entities.

Training: PPO/GAE or REINFORCE + value baseline, shared weights across seats, gamma=1, lr 3e-4, entropy 0.01, grad clip 1.0, batch 64–256 prefix decisions, 8 env workers. The 32-episode smoke proves signal, not win rate. Keep the 143k MLP as an ablation.

---

## 7. Later scale-up

**HET-full + history/league**:

- `d=192–256`, `L=6–8`, 8 heads, FFN 512–1024, ~3–8M params.
- Add per-trait tokens with parent embeddings and relation bias from `linked_entity_*`/`link_source`, or a GNN message-passing branch over explicit links.
- Add bounded public-history tokens (`K≈64`) and an auxiliary belief head over opponent hand composition. Train belief with privileged labels; anneal/oracle-guide only during training ([asymmetric actor-critic](https://arxiv.org/abs/1710.06542), [Suphx](https://arxiv.org/abs/2003.13590)).
- Training: PPO/IMPALA with V-trace for throughput; population/league/PSRO for 2–8 players; R-NaD-style regularization for 2p zero-sum ([DeepNash](https://arxiv.org/abs/2206.15378)). Search only if a public belief state is exposed (ReBeL/Student of Games).
- Perceiver/ISAB only if token count exceeds ~1k; at current N it adds bottleneck risk without throughput benefit. [Set Transformer](https://arxiv.org/abs/1810.00825) ISAB is the linear-attention option if needed.

---

## 8. Risks and evidence limits

- Rollout stats are small: 2 base-2p episodes, 2 all-2p episodes (one stalled), 1 all-8p episode; survival policy, not exhaustive. They size tensors, not prove distributions.
- Matrix status shows **194 errors / 1,418 episodes** (`rl/runs/matrix_00_status.json`); all-2p produced `ENGINE_STALLED: empty action domain for DEFENSE player_1`. Any architecture is blocked until action-domain coverage is fixed; do not mistake stalls for policy failure.
- Smoke report shows `gradient_fixture_error: "ValueError: The truth value of an array with more than one element is ambiguous"` and `gradient_fixture: null`; this is a learning-proof blocker, not an architecture issue.
- Parameter counts for proposed models are analytic estimates, not trained measurements. Sample-efficiency claims are inference from published relational/self-play results, not from this game.
- No repo files were edited; temporary measurement script was outside the repo.

**Key sources**

[Deep Sets](https://arxiv.org/abs/1703.06114) · [Set Transformer](https://arxiv.org/abs/1810.00825) · [Perceiver IO](https://arxiv.org/abs/2107.14795) · [Relation Networks](https://arxiv.org/abs/1706.01427) · [Graph Networks](https://arxiv.org/abs/1806.01261) · [Relational Deep RL](https://arxiv.org/abs/1806.01830) · [Pointer Networks](https://arxiv.org/abs/1506.03134) · [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z) · [SAINT](https://arxiv.org/abs/2505.12109) · [BDQ](https://arxiv.org/abs/1711.08946) · [Dulac-Arnold](https://arxiv.org/abs/1512.07679) · [NFSP](https://arxiv.org/abs/1603.01121) · [Deep CFR](https://arxiv.org/abs/1811.00164) · [ReBeL](https://arxiv.org/abs/2007.13544) · [DeepStack](https://arxiv.org/abs/1701.01724) · [DeepNash]…

[Some lines truncated to 768 chars. Use agent://ArchResearch:raw:222-222 to read line 222 whole]
