# Round 2's red-team checks

These are exploratory checks the round-2 critic ran before publication, kept as they ran. Only the
paths were made relative to the repository. Each uses 3 seeds or fewer, and none was
pre-registered. The README quotes them as exploratory. Each `.out` file is the run the README
cites.

| check | script | question |
|---|---|---|
| G1 | `r2_gate.py` | Does checking the recall against the cue (cue–read-out overlap) separate imagining from remembering? |
| G2 | `r2_native.py` | Does the filing rule still organise the memory when run through the network's own read-out (W_gh · ReLU(H_a c)) instead of the exact coefficients? |
| G3 | `r2_t3noise.py` | Are the theory's worst E4 rows (the T3 failure) sampling noise? Each is re-measured over 20 cue draws. |
| G4 | `r2_construct.py` | How much of the self-filed memory's construction deficit is factor C, whose grouping the oracle gets for free? |
| G5 | `r2_repredict.py` | Does the frozen theory reproduce the hashed out-of-sample predictions exactly? |

Run any of them from the repository root, for example `uv run python docs/redteam/r2_gate.py`.
