# RL research reports

Background research for the headless RL stack (`rl/node`, `rl/python`). Written by four parallel read-only research agents; each report cites its sources and marks unsupported claims as inference.

| Report | Question | Headline |
|---|---|---|
| [`architecture.md`](architecture.md) | Which model architecture replaces the starter MLP? | Hierarchical entity-token Transformer (HET-lite: `d=128`, 2 layers, ~0.4–0.7M params) with a pointer scorer over the current prefix choices. Keep the autoregressive prefix and dynamic `Discrete(len(choices))`. |
| [`training.md`](training.md) | Which algorithm and self-play scheme? | PPO-clipped over the existing candidate scorer with shared seat weights, Monte-Carlo advantage, entropy bonus and snapshot/PFSP self-play. DouZero-style deep Monte-Carlo is the main alternative. AlphaZero/MuZero/NFSP are not the first step. |
| [`stack.md`](stack.md) | Custom stack or an existing framework? | Keep the custom AEC + candidate stack. TorchRL/RLlib/SB3 cannot represent per-decision dynamic action spaces without caps; TorchRL is at most an optional replay-buffer dependency. |
| [`evaluation.md`](evaluation.md) | How do we measure progress honestly? | Primary metric is expected terminal payoff vs a frozen opponent with balanced seats; pre-registered dev/test/confirmation seed blocks, paired seeds, bootstrap CIs, SPRT only as a gate. Current 8-game self-play eval is a reload check, not an evaluation. |

## Measured facts these reports rely on

- Observation contract: `rl/schema.json`; decisions are prefix steps over the current `choices`.
- Engine cost on this machine: base 2p ≈0.7 s/episode, 8p all-addons ≈90 s/episode single worker.
- Starter checkpoint: 143,234 parameters; 32-episode smoke trains in ≈78 s on CPU.
- Population sizes sampled from real rollouts: ≤36 animals, ≤65 traits, ≤128 hand cards, ≤126 choices per decision; mean prefix depth ≈1.0–1.16.

## Known gaps flagged by the reports

- `train_smoke` policy loss is summed while entropy is averaged, making the entropy coefficient effectively inert.
- The type embedding is shared across catalog namespaces whose ids are namespace-local (trait id 1 aliases card id 1).
- Hand-card choice rows do not carry the card's two trait types, so the scorer sees only card type and hand index.
- The full addon matrix still reports engine errors; fix those before large-scale training.
