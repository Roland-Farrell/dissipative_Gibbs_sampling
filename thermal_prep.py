"""
Dissipative thermal-state preparation by statevector trajectory simulation.

Protocol (one reset cycle):
    1. initialize the bath in a thermal computational-basis state ~ exp(-beta H_E),
    2. evolve system+bath under a Trotterized H_tot = H_S + H_E + alpha*(O_S X_E),
    3. measure and reset the bath; keep the system state.
Iterating this channel converges the system toward the Gibbs state of H_S.

Switching Hamiltonians
----------------------
Everything below the `MODELS` section is Hamiltonian-agnostic. To run a different
system Hamiltonian you only add a `Model` in the MODELS section: define
its operator (a SparsePauliOp, used for the energy) and its `trotter_step`
(one second order Trotter step of H_S on the system qubits).

All Ising models (any topology) share the same second-order Trotter step -- an
Rzz layer on the bonds between Rx and Rz field layers -- so `ising_model` builds
both the operator and the step from just (N, edges, gx, gz); `kagome_ising` and
`tfim_chain` are one-liners on top of it.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator


# ============================================================================
# A coupling term is  coeff * X_bath (otimes) A_S, where A_S is a product of system
# Paulis. `sample` returns the (random) terms for one reset cycle as a list of
#   (paulis, bath_index, coeff)
# with paulis = [(pauli_char, system_site), ...] giving A_S. The coeffcient is
# chosen such that ||O_S||_2=1
@dataclass
class Coupling:
    name: str
    sample: Callable[[int, int, np.random.Generator],
                     List[Tuple[List[Tuple[str, int]], int, float]]]
    size: Optional[Callable[[int], int]] = None
    # size(N) -> number of distinct jump operators this coupling draws from, given
    # N system qubits. Used by `mix_couplings` to sample uniformly over the union.


# ============================================================================
# Model: the Hamiltonian-specific object.
# ============================================================================
@dataclass
class Model:
    name: str
    N: int                                  # number of system qubits
    hamiltonian: SparsePauliOp              # H_S (defines the energy observable)
    trotter_step: Callable[[QuantumCircuit, float], None]
    # trotter_step(circ, dt): apply one symmetric 2nd-order Trotter step of H_S
    # to system qubits 0..N-1, for time dt. Must be consistent with `hamiltonian`.
    coupling: Optional[Coupling] = None     # default (None) -> pauli_coupling()
    initial_state: Optional[Callable[[int, np.random.Generator], np.ndarray]] = None
    # initial_state(N, rng) -> 2^N statevector; default (None) -> random product state.


# ============================================================================
# MODELS  --  add a factory here to switch Hamiltonians; nothing else changes.
# ============================================================================
def kagome_edges(Lx, Ly, pbc=False):
    """Bond list of the (Lx, Ly) kagome lattice."""
    neigh = {0: [(0, 0, 1), (0, 0, 2), (-1, 0, 1), (0, -1, 2)],
             1: [(0, 0, 0), (0, 0, 2), (1, 0, 0), (1, -1, 2)],
             2: [(0, 0, 0), (0, 0, 1), (0, 1, 0), (-1, 1, 1)]}
    sid = lambda i, j, s: i + Lx * j + (Lx * Ly) * s
    edges = set()
    for i in range(Lx):
        for j in range(Ly):
            for s in range(3):
                for di, dj, sp in neigh[s]:
                    ii, jj = i + di, j + dj
                    if pbc:
                        ii, jj = ii % Lx, jj % Ly
                    elif not (0 <= ii < Lx and 0 <= jj < Ly):
                        continue
                    a, b = sid(i, j, s), sid(ii, jj, sp)
                    edges.add((min(a, b), max(a, b)))
    return sorted(edges)


def ising_model(N, edges, gx, gz, name="ising"):
    """Ising model on an arbitrary topology:
        H = sum_<ij in edges> Z_i Z_j  +  gx sum_i X_i  +  gz sum_i Z_i.

    Every Ising model Trotterizes the same way, so all you supply is the qubit
    count, the bond list, and the two field strengths -- this returns a ready
    `Model`. `kagome_ising` / `tfim_chain` just hand it different edge lists.
    """
    terms = ([("ZZ", [a, b], 1.0) for a, b in edges]
             + [("X", [i], gx) for i in range(N)]
             + [("Z", [i], gz) for i in range(N)])
    H = SparsePauliOp.from_sparse_list(terms, num_qubits=N)

    def trotter_step(circ, dt):
        # symmetric split of H_S for time dt:  e^{-i gx X dt/2} e^{-i ZZ dt} e^{-i gz Z dt} e^{-i gx X dt/2}
        circ.rx(gx * dt, range(N))
        for a, b in edges:
            circ.rzz(2 * dt, a, b)          # rzz(theta) = e^{-i theta/2 ZZ} -> e^{-i dt ZZ}
        circ.rz(2 * gz * dt, range(N))
        circ.rx(gx * dt, range(N))

    return Model(name, N, H, trotter_step)


def kagome_ising(Lx, Ly, gx, gz, pbc=False):
    return ising_model(3 * Lx * Ly, kagome_edges(Lx, Ly, pbc), gx, gz,
                       name=f"kagome_ising_{Lx}x{Ly}")


def tfim_chain(N, gx, gz, pbc=True):
    edges = [(n, (n + 1) % N) for n in range(N if pbc else N - 1)]
    return ising_model(N, edges, gx, gz, name=f"tfim_chain_{N}")


def _heis_bond(circ, a, b, theta):
    """exp(-i theta (X_a X_b + Y_a Y_b + Z_a Z_b)) -- the three terms commute."""
    circ.rxx(2 * theta, a, b)
    circ.ryy(2 * theta, a, b)
    circ.rzz(2 * theta, a, b)


def _edge_color_k(edges, k, budget=2_000_000):
    """Try to properly edge-color `edges` with exactly `k` colors by backtracking
    (most-constrained edge first). Returns {edge: color} or None if infeasible
    within `budget` search nodes."""
    used = {v: set() for e in edges for v in e}      # vertex -> colors on its edges
    color = {}
    remaining = list(edges)
    calls = [0]

    def backtrack():
        if not remaining:
            return True
        calls[0] += 1
        if calls[0] > budget:
            return False
        # pick the uncolored edge with the fewest available colors (fail fast)
        i = min(range(len(remaining)),
                key=lambda j: k - len(used[remaining[j][0]] | used[remaining[j][1]]))
        e = remaining.pop(i)
        a, b = e
        for c in range(k):
            if c not in used[a] and c not in used[b]:
                used[a].add(c); used[b].add(c); color[e] = c
                if backtrack():
                    return True
                used[a].discard(c); used[b].discard(c); del color[e]
        remaining.append(e)
        return False

    return dict(color) if backtrack() else None


def edge_coloring(edges):
    """Proper edge coloring into the *minimum* number of color classes (each a set
    of pairwise-disjoint bonds, which therefore commute). By Vizing's theorem the chromatic index 
    is Delta or Delta+1, where Delta is the maximum degree of a node; we find an optimal
    coloring by backtracking, trying Delta first. A Delta-regular graph on an odd
    number of vertices is necessarily class 2, so that case skips straight to Delta+1.
    Returns a list of bond-lists (the color layers). For kagome PBC this gives 4
    colors when N=3*Lx*Ly is even and 5 when odd."""
    edges = [tuple(sorted(e)) for e in edges]
    verts = {v for e in edges for v in e}
    deg = {v: 0 for v in verts}
    for a, b in edges:
        deg[a] += 1; deg[b] += 1
    Delta = max(deg.values()) if deg else 0
    regular = len(set(deg.values())) <= 1
    k_lo = Delta + 1 if (regular and len(verts) % 2 == 1) else Delta

    coloring = None
    for k in range(k_lo, Delta + 2):                  # Delta then (if needed) Delta+1
        coloring = _edge_color_k(edges, k)
        if coloring is not None:
            break
    groups = {}
    for e, c in coloring.items():
        groups.setdefault(c, []).append(e)
    return [sorted(groups[c]) for c in sorted(groups)]


def heisenberg_model(N, edges, J=1.0, name="heisenberg", *,
                     coupling=None, initial_state=None):
    """Isotropic Heisenberg model on an arbitrary topology:
        H = J sum_<ij in edges> (X_i X_j + Y_i Y_j + Z_i Z_j).

    The Trotter step is the canonical symmetric one over an edge coloring:
    neighboring bonds do not commute, so the bonds are grouped into disjoint
    (commuting) color layers and split as e^{-iH_1 dt/2} ... e^{-iH_c dt} ...
    e^{-iH_1 dt/2}.

    By default it uses the AFHM jump operators -- single-site Paulis
    and non-local (XX+YY+ZZ)/3 singlet pairs, sampled uniformly over their union
    -- and a singlet initial state; pass `coupling`/
    `initial_state` to override either.
    """
    terms = []
    for a, b in edges:
        terms += [("XX", [a, b], J), ("YY", [a, b], J), ("ZZ", [a, b], J)]
    H = SparsePauliOp.from_sparse_list(terms, num_qubits=N)

    colors = edge_coloring(edges)               # disjoint (commuting) bond layers

    def trotter_step(circ, dt):                 # symmetric 2nd order over color layers
        for group in colors:
            for a, b in group:
                _heis_bond(circ, a, b, J * dt / 2)
        for group in reversed(colors):
            for a, b in group:
                _heis_bond(circ, a, b, J * dt / 2)

    if coupling is None:
        coupling = mix_couplings([pauli_coupling(), heisenberg_coupling()])
    if initial_state is None:
        initial_state = singlet_initializer(dimer_cover(N, edges))
    return Model(name, N, H, trotter_step, coupling, initial_state)


def heisenberg_chain(N, J=1.0, pbc=True, **kw):
    """Isotropic Heisenberg on a 1D chain."""
    edges = [(n, (n + 1) % N) for n in range(N if pbc else N - 1)]
    return heisenberg_model(N, edges, J, name=f"heisenberg_chain_{N}", **kw)


# ============================================================================
# COUPLINGS  --  the system-bath jump operators.
# ============================================================================
def pauli_coupling():
    """Default single-site jump operators  O = {X, Y, Z}: each
    bath qubit couples via  O_site (otimes) X_bath, with O a random single-qubit Pauli
    on a distinct random site."""
    def sample(N, N_bath, rng):
        kinds = rng.choice(["X", "Y", "Z"], size=N_bath)
        sites = rng.choice(N, size=N_bath, replace=False)
        return [([(kinds[n], int(sites[n]))], n, 1.0) for n in range(N_bath)]
    return Coupling("pauli", sample, size=lambda N: 3 * N)   # 3 Paulis x N sites


def heisenberg_coupling(pairs=None):
    """SU(2)-singlet (S=0) jump operators:
        O_{i,j} = (X_iX_j + Y_iY_j + Z_iZ_j) / 3,   coupled as  O_{i,j} otimes X_bath.

    The 1/3 normalizes ||O||_2 = 1. Each bath qubit picks a system pair (i, j) -- any two
    distinct qubits by default, so the coupling can be non-local; pass `pairs`
    (a list of (i, j)) to restrict it. The three terms commute, so the exp
    factorizes exactly into three 3-qubit rotations."""
    def sample(N, N_bath, rng):
        terms = []
        for n in range(N_bath):
            if pairs is None:
                i, j = (int(x) for x in rng.choice(N, size=2, replace=False))
            else:
                i, j = pairs[int(rng.integers(len(pairs)))]
            for P in ("X", "Y", "Z"):
                terms.append(([(P, int(i)), (P, int(j))], n, 1.0 / 3.0))
        return terms
    size = (lambda N: N * (N - 1) // 2) if pairs is None else (lambda N: len(pairs))
    return Coupling("heisenberg", sample, size=size)         # all (unordered) pairs


def mix_couplings(couplings, weights=None):
    """Combine several couplings: each bath qubit independently picks one of
    `couplings` and uses its jump operator. With weights=None (default) the pick
    is weighted by each coupling's operator count `size(N)`, i.e. uniform over the
    *union* of all individual jump operators
    Pass `weights` (summing to 1) to override with fixed per-coupling probabilities."""
    def sample(N, N_bath, rng):
        w = weights
        if w is None and all(c.size is not None for c in couplings):
            sizes = np.array([c.size(N) for c in couplings], dtype=float)
            w = sizes / sizes.sum()                          # uniform over all operators
        terms = []
        for n in range(N_bath):
            c = couplings[rng.choice(len(couplings), p=w)]
            for paulis, _, coeff in c.sample(N, 1, rng):     # sample 1 bath qubit
                terms.append((paulis, n, coeff))             # relabel to bath qubit n
        return terms
    return Coupling("mix(" + "+".join(c.name for c in couplings) + ")", sample)


# ============================================================================
# INITIAL STATES
# ============================================================================
def dimer_cover(N, edges):
    """Greedy matching of `edges`: a set of disjoint bonds."""
    used, cover = set(), []
    for a, b in edges:
        if a not in used and b not in used:
            cover.append((a, b)); used |= {a, b}
    return cover


def singlet_state(N, bonds):
    """Statevector for a tensor product of SU(2) singlets (|01>-|10>)/sqrt(2)
    on the given disjoint `bonds`; any unpaired qubit is left in |0>."""
    qc = QuantumCircuit(N)
    for a, b in bonds:
        qc.h(a); qc.cx(a, b); qc.x(b); qc.z(a)      # -> (|01>-|10>)/sqrt(2) on (a,b)
    return Statevector(qc).data


def singlet_initializer(bonds):
    """initial_state callable that returns the singlet state."""
    def init(N, rng):
        return singlet_state(N, bonds).copy()
    return init


# ============================================================================
# Generic machinery (Hamiltonian-independent).
# ============================================================================
def _gaussian_envelope(nTrot, t_tot, sigma=1e2):
    dt = t_tot / nTrot
    xs = np.arange(-t_tot / 2 + t_tot / (2 * nTrot), t_tot / 2, t_tot / nTrot)
    g = np.exp(-xs ** 2 / t_tot ** 2 / (4 * sigma ** 2))
    return g / np.sqrt(dt * np.sum(g ** 2))


def _pauli_rotation(circ, pauli_qubits, theta):
    """exp(-i theta/2 * P), P = product of the given (pauli_char, qubit) factors.
    Standard basis-change + CX-ladder + Rz construction (a generalized rzz)."""
    qs = [q for _, q in pauli_qubits]
    for p, q in pauli_qubits:                        # rotate each factor into the Z basis
        if p == "X":
            circ.h(q)
        elif p == "Y":
            circ.sdg(q); circ.h(q)
    for i in range(len(qs) - 1):
        circ.cx(qs[i], qs[i + 1])
    circ.rz(theta, qs[-1])                            # rz(theta) = exp(-i theta/2 Z)
    for i in reversed(range(len(qs) - 1)):
        circ.cx(qs[i], qs[i + 1])
    for p, q in pauli_qubits:                         # undo the basis change
        if p == "X":
            circ.h(q)
        elif p == "Y":
            circ.h(q); circ.s(q)


def _add_evolution(circ, model, omega, alpha, nTrot, t_tot, rng, coupling, sigma=1e2):
    """One full Trotterized H_tot: coupling | H_E | H_S | H_E | coupling, per Trotter step.

    The coupling terms (which Paulis on which sites) are drawn once for the whole
    cycle; only the Gaussian-envelope amplitude changes between sub-steps."""
    N, N_bath = model.N, len(omega)
    dt = t_tot / nTrot
    g = _gaussian_envelope(nTrot, t_tot, sigma)
    terms = coupling.sample(N, N_bath, rng)           # [(paulis, bath_index, coeff), ...]

    def coupling_layer(st):
        amp = g[st] * dt * alpha                       # = g[st] * dt * 2 * alpha / 2
        for paulis, b, coeff in terms:
            # term = coeff * X_bath (otimes) prod(paulis), applied as one Pauli rotation
            _pauli_rotation(circ, [("X", N + b)] + list(paulis), coeff * amp)

    for st in range(nTrot):
        coupling_layer(st)
        for n in range(N_bath):
            circ.rz(-dt * omega[n] / 2, N + n)         # H_E = sum omega_n/2 Z_n
        model.trotter_step(circ, dt)                   # <-- the only model-specific call
        for n in range(N_bath):
            circ.rz(-dt * omega[n] / 2, N + n)
        coupling_layer(st)


def _sample_thermal_bath(omega, beta, rng):
    """Bath as a thermal computational-basis sample; returns the bath integer index.
    P(qubit n excited) = 1/(1+exp(beta*omega_n))."""
    p_exc = 1.0 / (1.0 + np.exp(np.clip(beta * omega, -700, 700)))
    excited = rng.random(len(omega)) < p_exc
    bath_int = 0
    for n, e in enumerate(excited):
        if e:
            bath_int |= (1 << n)                              # bath qubit n -> bit n
    return bath_int


def _random_product_state(N, rng):
    bits = rng.integers(0, 2, size=N)
    idx = 0
    for b in bits:
        idx = (idx << 1) | int(b)
    vec = np.zeros(1 << N, dtype=complex)
    vec[idx] = 1.0
    return vec


def _embed(sys_state, bath_int, N, N_bath):
    """Full state = |bath_int>_bath (otimes) |sys_state>_system  (bath = most significant qubits)."""
    full = np.zeros(1 << (N + N_bath), dtype=complex)
    full[bath_int << N: (bath_int + 1) << N] = sys_state
    return full


def _collapse_system(full_sv, N, N_bath, rng):
    """Sample a bath measurement outcome and return the normalized post-measurement system state."""
    m = full_sv.reshape((1 << N_bath, 1 << N))               # rows = bath, cols = system
    probs = (np.abs(m) ** 2).sum(axis=1)
    probs /= probs.sum()
    o = rng.choice(1 << N_bath, p=probs)
    s = m[o].copy()
    return s / np.linalg.norm(s)


def _expval(H, state):
    return float(np.vdot(state, H @ state).real)


def thermal_ensemble(H, beta, cutoff_over_beta=8):
    """Low-energy Gibbs ensemble of H_S.

    Returns (eigvecs, weights): the eigenstates within cutoff_over_beta/beta of
    the ground state (columns of eigvecs, shape (2^N, K)) and their normalized
    Boltzmann weights w_i ~ exp(-beta E_i) (shape (K,)). Used to build the
    target state sigma = sum_i w_i |e_i><e_i| for the fidelity.
    """
    H = H.toarray() if hasattr(H, "toarray") else np.asarray(H)
    evals, evecs = np.linalg.eigh(H)                  
    mask = (evals - evals[0]) < cutoff_over_beta / beta
    w = np.exp(-beta * (evals[mask] - evals[0]))
    w /= w.sum()
    return evecs[:, mask], w


def hs_fidelity(states, eigvecs, weights):
    """Normalized Hilbert-Schmidt overlap between the sampled ensemble
        rho = (1/M) sum_j |psi_j><psi_j|
    and the target Gibbs state
        sigma = sum_i w_i |e_i><e_i|:

        F = Tr[rho sigma] / max(Tr[rho^2], Tr[sigma^2]).

    Tr[rho^2] uses the unbiased, i.e.
    it estimates the purity of the *underlying* distribution, not of the finite
    sample. Returns (F, stderr).
    """
    V = np.asarray(states)                            # (M, 2^N), rows = statevectors
    M = V.shape[0]
    overlap = np.abs(V @ eigvecs.conj()) ** 2 @ weights   # o_j = <psi_j| sigma |psi_j>
    num = float(overlap.mean())                       # = Tr[rho sigma]
    G = V @ V.conj().T                                 # Gram matrix, |G_jk|^2 = |<psi_j|psi_k>|^2
    purity_rho = (np.sum(np.abs(G) ** 2) - M) / (M ** 2 - M)   # unbiased Tr[rho^2]
    purity_sigma = float(np.sum(weights ** 2))        # Tr[sigma^2]
    F = num / max(purity_rho, purity_sigma)
    stderr = float(overlap.std() / np.sqrt(M) / purity_sigma)
    return F, stderr


def _auto_batch(n_total_qubits, nSamples, budget_bytes=8_000_000_000):
    """Largest batch of circuits whose statevectors fit the memory budget. Default 8Gb"""
    per = (1 << n_total_qubits) * 16 * 2                      # ~ stored state + result, complex128
    return max(1, min(nSamples, budget_bytes // per))



def run_thermal_prep(model, *, alpha, beta, t_tot, nTrot, nSteps, nSamples,
                     N_bath, omega_max, sigma=1e2, coupling=None, init_state=None,
                     batch_size=None, seed=None, progress=True,
                     compute_fidelity=False, fidelity_cutoff=8):
    """
    Returns energies, shape (nSteps+1, nSamples): row 0 is the initial energy,
    row k the energy after k reset cycles, per trajectory. Divide by model.N for density.

    Each cycle draws a fresh bath frequency per qubit, omega ~ U[0, omega_max].
    sigma      : width of the Gaussian filter f(t) = N exp(-t^2/(4 sigma^2 T^2))
                 (Eq. 10). Large sigma (default 1e2) -> effectively constant
                 filter; sigma ~ 1/4 smoothly turns the coupling on/off, which
                 lowers the fixed-point error at longer T.
    coupling   : system-bath jump operators; default falls back to
                 model.coupling, else pauli_coupling().
    init_state : initial-state callable (N, rng) -> statevector; default falls
                 back to model.initial_state, else a random product state.
    batch_size : circuits per Aer job (default: memory-budgeted auto value).

    If compute_fidelity=True, also returns the normalized Hilbert-Schmidt
    fidelity of the sampled ensemble against the Gibbs state of model.hamiltonian
    at this beta (see `hs_fidelity`). In that case the return value is
        (energies, fidelity, fidelity_stderr)
    where fidelity and fidelity_stderr have shape (nSteps+1,).
    """
    rng = np.random.default_rng(seed)
    N = model.N
    H = model.hamiltonian.to_matrix(sparse=True)             # built once
    backend = AerSimulator(method="statevector")
    if batch_size is None:
        batch_size = _auto_batch(N + N_bath, nSamples)
    coupling = coupling or model.coupling or pauli_coupling()
    init_state = init_state or model.initial_state or _random_product_state

    if compute_fidelity:
        eigvecs, weights = thermal_ensemble(model.hamiltonian.to_matrix(sparse=True),
                                            beta, fidelity_cutoff)
        fidelity = np.zeros(nSteps + 1)
        fidelity_err = np.zeros(nSteps + 1)

    states = [init_state(N, rng) for _ in range(nSamples)]
    energies = np.zeros((nSteps + 1, nSamples))
    energies[0] = [_expval(H, s) for s in states]
    if compute_fidelity:
        fidelity[0], fidelity_err[0] = hs_fidelity(states, eigvecs, weights)

    for step in range(nSteps):
        circuits = []
        for s in range(nSamples):
            omega = rng.uniform(0, omega_max, size=N_bath)
            bath_int = _sample_thermal_bath(omega, beta, rng)
            circ = QuantumCircuit(N + N_bath)
            circ.set_statevector(Statevector(_embed(states[s], bath_int, N, N_bath)))
            _add_evolution(circ, model, omega, alpha, nTrot, t_tot, rng, coupling, sigma)
            circ.save_statevector()
            circuits.append(circ)

        # submit independent trajectories in memory-bounded batches (one job each)
        svs = [None] * nSamples
        for lo in range(0, nSamples, batch_size):
            chunk = circuits[lo:lo + batch_size]
            res = backend.run(chunk).result()
            for k in range(len(chunk)):
                svs[lo + k] = np.asarray(res.get_statevector(k))

        for s in range(nSamples):
            states[s] = _collapse_system(svs[s], N, N_bath, rng)
            energies[step + 1, s] = _expval(H, states[s])

        if compute_fidelity:
            fidelity[step + 1], fidelity_err[step + 1] = hs_fidelity(states, eigvecs, weights)

        if progress:
            msg = f"step {step + 1}/{nSteps}  <H>/N = {energies[step + 1].mean() / N:+.4f}"
            if compute_fidelity:
                msg += f"   F = {fidelity[step + 1]:.4f}"
            print(msg, flush=True)

    if compute_fidelity:
        return energies, fidelity, fidelity_err
    return energies


# ============================================================================
# Example run for (2,2) Kagome Ising model
# ============================================================================
if __name__ == "__main__":
    gx, gz = 0.5, 2.0
    model = kagome_ising(Lx=2, Ly=2, gx=gx, gz=gz, pbc=False)  # <-- swap this line for another Model
    energies = run_thermal_prep(
        model, alpha=1.75, beta=100.0, t_tot=0.75, nTrot=3,
        nSteps=100, nSamples=100, N_bath=1, omega_max=max(4 * gx, 4 * gz, 4),
        seed=0, compute_fidelity=False,
    )
    Ess = energies[-1].mean() / model.N
    sem = energies[-1].std() / np.sqrt(energies.shape[1]) / model.N
    print(f"\n{model.name}: steady <H>/N = {Ess:.4f} +/- {sem:.4f}")
