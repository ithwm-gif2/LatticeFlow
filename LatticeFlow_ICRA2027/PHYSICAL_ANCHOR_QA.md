# Physical-Anchor Manuscript Addendum

## Claim audit

- Supported: all 15 physical-anchor source positions are distinct in the internal Flow state.
- Supported: both source variants reduce held-out selected cost and collision proxy relative to fair YOPO.
- Not supported: physical-anchor Flow uniformly outperforms residual-source Flow.
- Supported only descriptively: physical-anchor selected cost is tied with residual source; selection regret is lower, while clearance and collision proxy are worse.
- Supported only in one map/five goals: both physical raw and selector policies achieve 5/5 success and 0/5 collision; the selector strongly reduces switching.

## Reviewer-facing interpretation

The physical representation resolves the mathematical criticism that all residual-space sources are zero. However, the paired ablation also shows that explicit source separation is not itself sufficient for better navigation metrics. This negative result should remain in the abstract, results, and conclusion because it narrows the causal claim and prevents the paper from attributing cost-refinement or selector gains to source coordinates.

## Remaining high-priority gaps

1. Same-backbone direct regression trained on identical refined targets.
2. Multiple training seeds for both source variants.
3. Multiple ROS map seeds and non-ceiling collision/success conditions.
4. Dynamic-obstacle evaluation.
5. Real-quadrotor results and platform details.

The independent statistical unit is a complete rollout for ROS and a map for broad environmental generalization. Query-level bootstrap intervals are descriptive because the 20,000 observations are nested within two held-out maps.
