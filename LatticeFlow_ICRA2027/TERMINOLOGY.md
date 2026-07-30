# Terminology Ledger

| Canonical term | Definition | Avoided variants | Decision |
|---|---|---|---|
| LatticeFlow | cost-refined flow matching over physical motion primitives | YOPO-FM, flow YOPO | Use `LatticeFlow` throughout. |
| motion-primitive lattice | fixed $3\times5$ camera-aligned bank inspired by prior spherical-lattice planning | generic grid | Name the prior method only once where inspiration is declared. |
| physical flow state | normalized body-frame terminal position, velocity, and acceleration | latent offset | Denote by $z^\ell$. |
| primitive residual | normalized yaw, pitch, radius, velocity, and acceleration adjustment used by the trajectory decoder | YOPO residual | Denote by $r^\ell$ without attributive naming. |
| explicit physical lattice source | $z_0^\ell=[p_{\rm anc}^\ell/s_p,0,0]$ | zero source | Emphasize that all 15 source vectors are physically distinct. |
| cost-refined target | detached anchor/student/local candidate after monotonic privileged-cost refinement | label, ground truth | The target is a policy-improvement target, not demonstrated ground truth. |
| direct lattice regression | retrained one-stage baseline on the identical map split | fair YOPO | Main-text baseline name. |
| teacher-guided diagnostic | flow variant using frozen direct-policy proposals and backbone initialization | main method | Use only to bound the effect of teacher information. |
| privileged ESDF supervision | map/point-cloud distance queries available only during training | map input | Keep the train/deployment boundary explicit. |
| perception-domain matching | native MID-360 FOV plus recorded return-mask preprocessing matched in training and inference | generic domain randomization | Separate it from the core standard-view benchmark. |
| virtual ceiling | sparse synthetic depth returns from a world-$z$ plane | altitude constraint | It is an image-space sensing prior, never a trajectory hard constraint. |
| local consistency regularization | output consistency under small depth and ego-state perturbations | temporal loss | Training samples are independent. |
| continuity-aware selection | optional post-network hysteresis using one previous endpoint | history policy | State that the real flights disable it. |
| collision proxy | held-out query whose projected clearance is below 0.6 m | collision rate | Never equate with physical collision. |
| real-flight success | final odometry within 1 m of goal without observed contact | success rate | With two trials, report counts and descriptive metrics only. |

## Locked notation

- $D_t$: current normalized single-frame depth image.
- $s_t=[v_t,a_t,g_t]$: body-frame velocity, acceleration, and goal direction.
- $\ell\in\{1,\dots,15\}$: lattice cell.
- $r^\ell$: normalized primitive residual.
- $\psi_\ell(r^\ell)=[p_T,v_T,a_T]$: residual-to-physical terminal-state transform.
- $z^\ell=S\psi_\ell(r^\ell)$: normalized physical flow coordinate.
- $z_0^\ell$: explicit physical source for cell $\ell$.
- $r_1^\ell$: cost-refined residual target; $z_1^\ell=S\psi_\ell(r_1^\ell)$.
- $v_\theta(z,\tau,c)$: conditional physical-state vector field.
- $J$: differentiable trajectory cost.
- $K$: Euler integration steps / neural-function evaluations.

## Claim boundary

- Supported: teacher-free training from random initialization improves held-out proxies over the retrained direct lattice baseline and is feasible onboard.
- Supported narrowly: the optional selector reduces switching in five rollouts on one simulated forest.
- Feasibility only: two real flights complete without observed collision; this is not a statistically estimated success rate.
- Not established: flow is superior to a same-backbone direct regressor trained on identical self-refined targets, robustness across training seeds, or dynamic-obstacle performance.
