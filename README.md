# Dissipative thermal-state preparation (statevector)

A statevector implementation of the dissipative quantum Gibbs
sampler originally developed in *"End-to-End Efficient Quantum Thermal and Ground State Preparation Made Simple"* (https://arxiv.org/abs/2508.05703) and demonstrated on IBM's quantum computers in *"Preparing thermal states of frustrated quantum spin systems using 139 qubits"* (https://arxiv.org/abs/2605.26245).
Notation and equations will refer to 2605.26245.  The algorithm prepares approximate Gibbs states `ρ_S(β) = e^{-βH_S}/Z` by repeatedly coupling the system to a freshly-reset thermal environment, and measures observables (energy, fidelity) by averaging over
trajectories.

Everything lives in [`thermal_prep.py`](thermal_prep.py).

## The algorithm (paper Algorithm 1)

Each reset cycle applies the channel `Φ(ρ_S) = Tr_E[U(T)(ρ_S ⊗ ρ_E)U†(T)]`:

1. sample the environment in a thermal product state `ρ_E(β) ∝ e^{-βH_E}`,
2. evolve system+environment under a Trotterized `H(t) = H_S + H_E + α f(t) H_SE`,
3. measure (collapse) the environment qubits.

Iterating `nSteps` times cools the system toward `ρ_S(β)`.

## Install

```bash
pip install qiskit qiskit-aer numpy
```

## Quick start

```python
from thermal_prep import kagome_ising, run_thermal_prep
import numpy as np

gx, gz = 0.5, 2.0
model = kagome_ising(Lx=2, Ly=2, gx=gx, gz=gz, pbc=False)     # N_S = 12
energies, F, F_err = run_thermal_prep(
    model, alpha=1.75, beta=100.0, t_tot=0.75, nTrot=3,
    nSteps=20, nSamples=200, N_bath=1, omega_max=max(4*gx, 4*gz, 4),
    seed=0, compute_fidelity=True,
)
print("steady <H>/N =", energies[-1].mean() / model.N)
print("fidelity     =", F[-1])
```

`run_thermal_prep` returns `energies` of shape `(nSteps+1, nSamples)`
(divide by `model.N` for the density); with `compute_fidelity=True` it also
returns the per-step fidelity and its standard error.

## Switching the Hamiltonian

A `Model` bundles the energy observable (`hamiltonian`), one symmetric Trotter
step (`trotter_step`), and optional defaults for the `coupling` and
`initial_state`. Built-in models

```python
from thermal_prep import ising_model, kagome_ising, tfim_chain, \
                          heisenberg_model, heisenberg_chain

ising_model(N, edges, gx, gz)        # AFIM on any topology
kagome_ising(Lx, Ly, gx, gz, pbc)    # AFIM on the kagome lattice
tfim_chain(N, gx, gz, pbc)           # AFIM on a 1D chain

heisenberg_model(N, edges, J)        # AFHM on any topology
heisenberg_chain(N, J, pbc)          # AFHM on a 1D chain
```

All Ising models share one Trotter step (an `Rzz` layer on the bonds between
`Rx`/`Rz` field layers). Heisenberg models Trotterize over an **edge coloring**
of the lattice (disjoint, commuting bond layers), since neighboring `XX+YY+ZZ`
bonds do not commute. To add a brand-new model, just build a `SparsePauliOp` and
a `trotter_step(circ, dt)` and wrap them in `Model`.

## Switching the jump operators (couplings)

The system-environment coupling `H_SE = Σ O_{i} ⊗ X_{bath}` (Eq. 9) is a
`Coupling` object. Built-ins:

```python
from thermal_prep import pauli_coupling, heisenberg_coupling, mix_couplings

pauli_coupling()           # O ∈ {X, Y, Z} single-site         (Eq. 9; AFIM default)
heisenberg_coupling()      # O = (XX+YY+ZZ)/3 on any pair       (Eq. 17; non-local, S=0)
heisenberg_coupling(pairs=[(0,1),(2,3)])   # restrict to given pairs
mix_couplings([pauli_coupling(), heisenberg_coupling()])   # mixed couplings for AFHM
```

All operators satisfy `‖O‖₂ = 1`. The AFHM samples jump operators **uniformly
over the union** of the `3N` single-site Paulis and the `N(N-1)/2` singlet pairs
(so `P(singlet) = N(N-1)/2 / (N(N-1)/2 + 3N)`).
`mix_couplings` with `weights=None` reproduces this exactly by weighting each
coupling by its operator count, and it is the **default** for `heisenberg_model`.
Override per run with `run_thermal_prep(..., coupling=...)` or set it on the
model; pass explicit `weights` to `mix_couplings` for fixed per-type probabilities.

## Switching the initial state

```python
from thermal_prep import singlet_initializer, dimer_cover
```

- **AFIM:** random computational-basis product state (`β = 0`) — the default.
- **AFHM:** tensor product of nearest-neighbor SU(2) singlets, the default for
  `heisenberg_model` (built from `dimer_cover(N, edges)`). Override with
  `run_thermal_prep(..., init_state=singlet_initializer(my_bonds))`.

## Convention map (code ↔ paper)

| Paper | Code | Notes |
|-------|------|-------|
| `H_AFIM = Σ Z_iZ_j + Σ(g_xX_i + g_zZ_i)` (Eq. 1) | `ising_model` | |
| `H_AFHM = Σ(X_iX_j+Y_iY_j+Z_iZ_j)` (Eq. 1) | `heisenberg_model` (`J=1`) | |
| `H_E = -½ Σ ω_i Z_i` (Eq. 6) | `rz(-dt·ω/2)` ×2 per step | applied as two half-steps |
| `Pr(X_{iE}) = 1/(1+e^{βω})` (Eq. 7) | `_sample_thermal_bath` | |
| `ω ∈ (0, ω_max]`, `ω_max = max(4g_x,4g_z,4)` | `omega_max`, uniform draw | `=8` for the kagome AFIM |
| `O ∈ {X,Y,Z}` (Eq. 9) | `pauli_coupling` | |
| `O = (XX+YY+ZZ)/3` (Eq. 17) | `heisenberg_coupling` | the `/3` gives `‖O‖₂=1` |
| AFHM jumps: uniform over `{Paulis} ∪ {singlet pairs}` | `mix_couplings([pauli_coupling(), heisenberg_coupling()])` | default for `heisenberg_model` |
| `f(t) = N·exp(-t²/4σ²T²)` (Eq. 10), `Σδt f²=1` (Eq. 12) | `_gaussian_envelope`, `run_thermal_prep(..., sigma=...)` | `σ→∞` (default `1e2`) ⇒ constant; `σ≈1/4` smoothly ramps the coupling on/off |
| `{δt, T, α} = {0.25, 0.75, 1.75}`, `T/δt = 3` | `t_tot=0.75, nTrot=3, alpha=1.75` | |
| `N_E` environment qubits | `N_bath` | paper uses `N_E=1` for statevector |
| singlet init `(|01>-|10>)/√2` | `singlet_initializer` | little-endian via circuit |

To reproduce the Section III lattices use `pbc=True` (`N_S = 12, 18, 24` for Kagome lattices
`(L_x,L_y) = (2,2),(2,3),(2,4)`).


## Performance notes
- Trajectories in a cycle are submitted as one **batched** Aer job (memory-budgeted).
- Fidelity requires a dense diagonalization of `H_S` (exponential in `N`), so it
  is only computed when `compute_fidelity=True`.
