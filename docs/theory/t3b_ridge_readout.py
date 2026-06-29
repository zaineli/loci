"""Section 3 extension: a ridge read-out W_sh = S H^T (H H^T + lam I)^-1 trades the leverage variance for
bias. Composition at empty addresses, and the clean read-out overlap at stored addresses, vs lam."""
from t3_compose import *

def run_lam(seed, P, lams):
    content = factored(DIM, P, np.random.default_rng(10_000 * seed + P))
    S, F = content.patterns, content.factors; A, B, Cc = content.codes
    grid = Scaffold(seed=seed); where = place.oracle(grid, F); Ha = grid.H[:, where]
    stored = set(map(tuple, F)); taken = np.zeros(grid.addresses, bool); taken[where] = True
    tests = []
    for a in range(9):
        for b in range(16):
            for c in range(5):
                if (a, b, c) in stored: continue
                for slot in range(5):
                    addr = grid.index(np.array([[a, b, c * 5 + slot]]))[0]
                    if not taken[addr]: tests.append(((a, b, c), addr)); break
    tests = tests[:150]
    scale = np.trace(Ha @ Ha.T) / Ha.shape[0]
    out = []
    for lam in lams:
        W = S @ Ha.T @ np.linalg.inv(Ha @ Ha.T + lam * scale * np.eye(Ha.shape[0])) if lam > 0 else S @ np.linalg.pinv(Ha)
        comp = np.mean([np.mean(np.sign(W @ grid.H[:, addr]) * np.sign(A[:, a] + B[:, b] + Cc[:, c])) for (a, b, c), addr in tests])
        own = np.mean(np.sign(W @ Ha) * S)                      # clean read-out at stored addresses
        proto = np.mean(np.sign(W @ Ha) * np.sign(A[:, F[:, 0]] + B[:, F[:, 1]] + Cc[:, F[:, 2]]))
        out.append((comp, own, proto))
    return np.array(out)

lams = (0, 0.01, 0.03, 0.1, 0.3, 1.0)
for P in (400, 800):
    res = np.mean([run_lam(s, P, lams) for s in (0, 1, 2)], 0)
    for lam, (comp, own, proto) in zip(lams, res):
        print(f"P {P} lam/mean-eig {lam:5}: composition at empty cells {comp:.3f}   stored read-out vs item {own:.3f}   vs its composite {proto:.3f}")
