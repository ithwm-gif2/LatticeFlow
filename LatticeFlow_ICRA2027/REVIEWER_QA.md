# Historical pre-real-flight review

> This review was produced before the two real-flight bags were analyzed. The current manuscript now includes two collision-free LiDAR-only feasibility flights, sensor-domain ablations, and onboard timing. Comments about `TBD` real-world fields are therefore superseded; concerns about the small real-flight sample, single training seed, one ROS map, and missing same-backbone direct-regression control remain valid.

# Review setup

- Input scope: complete six-page double-anonymous ICRA-style manuscript, method figures, held-out offline results, two learned ablations, NFE/latency study, and 15 matched ROS rollouts across three policy variants.
- Assessment boundary: no physical-flight result is available; closed-loop evidence contains five goals in one static random-forest map; learned runs use one training seed.
- Shared manuscript claim summary: a deterministic YOPO spherical lattice can serve as a structured flow source; teacher/student cost refinement improves trajectory endpoints; an optional one-step selector reduces closed-loop lattice switching without adding image or action history to the neural network.
- Visible evidence base: leakage-controlled map split, fair retrained YOPO, 20,000 held-out queries, target and consistency ablations, NFE/latency measurements, and matched rollout-level ROS metrics.
- Missing materials affecting confidence: repeated training seeds, multiple closed-loop map seeds, real-quadrotor trials, dynamic obstacles, and a same-backbone direct-regression control trained on the same cost-refined targets.

# Reviewer 1

- Overall assessment: The implementation and leakage controls are stronger than the typical preliminary learning-planner study, and the manuscript reports negative or modest ablation outcomes honestly. The central engineering case is nevertheless incomplete because the current experiments do not isolate whether flow matching is necessary.
- Who would be interested in the results, and why: Researchers in learned local planning, generative robot policies, and agile aerial autonomy would value a low-latency bridge between motion primitives and flow objectives.
- Major strengths: complete-map train/test separation; a YOPO baseline retrained on the same maps; explicit distinction between offline proxy metrics and closed-loop outcomes; raw-argmin versus selector comparison; full failure denominators; and measured end-to-end network timing.
- Major concerns: The closest missing control is a same-backbone direct regressor trained against the identical cost-refined endpoint and score objectives. NFE=1 evaluation of a flow-trained model is not equivalent to training a direct regressor. The teacher-target ablation changes total cost by less than 0.05%, so the evidence for the target-search contribution is primarily a modest safety-proxy change. All ROS methods succeed without collision, preventing a closed-loop safety ranking.
- Technical failings that need to be addressed before the case is established: add the direct-regression control; repeat at least the main method over multiple training seeds; evaluate multiple independent simulator maps/seeds with more difficult trials that yield non-ceiling success and collision outcomes; and calibrate the depth collision proxy against geometric collision or clearance distributions.
- Assessment against Nature-style criteria: originality is plausible in the lattice-as-source construction, but its distinction from residual regression through a neural ODE needs the missing control. Scientific importance is currently field-local. Interdisciplinary reach is limited but appropriate for ICRA. Technical soundness is good for the reported scope, not for broad deployment claims. Readability is generally clear.
- Recommendation posture: promising and technically careful, but the flow-specific causal claim is not yet fully established.

# Reviewer 2

- Overall assessment: The paper has a coherent conceptual synthesis of A2A's informative-source principle and YOPO's geometric lattice. The novelty is most credible when framed as an engineering representation and training recipe, not as a new general theory of flow matching.
- Who would be interested in the results, and why: The work will interest researchers seeking structured generative priors for real-time control and those adapting diffusion/flow policies to platforms where random sampling is undesirable.
- Major strengths: the intellectual-debt boundary is explicit; multimodality is correctly attributed to the 15 cells rather than claimed as within-cell stochastic generation; the teacher is not presented as a guaranteed upper bound; and the paper reports that consistency regularization has only modest offline benefit.
- Major concerns: The main selected-cost gain relative to YOPO may arise from additional trajectory supervision, score learning, backbone/training differences, or iterative parameterization rather than flow transport itself. The continuity selector, rather than the learned flow, produces most of the ROS switch reduction. The paper should avoid implying that deterministic lattice initialization alone guarantees temporal smoothness.
- Technical failings that need to be addressed before the case is established: provide a component-matched regression baseline; separate the gains from target search, on-policy trajectory loss, flow integration, and post-network selection; and report seed variation. The use of the previous selected endpoint is legitimate, but should be described as one-step internal selector state so that “single-frame” is not read as a fully memoryless deployed system.
- Assessment against Nature-style criteria: originality is credible but incremental and must be scoped narrowly. Scientific importance rests on reliable real-time deployment, which still lacks real-flight evidence. Broader readership may appreciate structured sources for generative control, but the present evidence is quadrotor-specific. Technical soundness is acceptable within the stated simulation scope. The method figure materially improves accessibility.
- Recommendation posture: potentially strong ICRA contribution after a direct-regression comparison and broader closed-loop validation.

# Reviewer 3

- Overall assessment: The manuscript is unusually transparent about leakage, proxies, small ROS sample size, and incomplete real-world validation. Its main weakness for nonspecialist readers is a dense abstract that combines source design, target refinement, regularization, and selector behavior before explaining which component actually drives each measured gain.
- Who would be interested in the results, and why: Beyond quadrotor planning, readers working on structured priors for robot action generation may find the lattice-source idea reusable when an action space already has a physical discretization.
- Major strengths: clear input/privileged-information boundary; concise schematic; explicit negative evidence from ablations; and a useful safety-versus-continuity trade-off rather than a one-sided performance claim.
- Major concerns: The distinction between neural-network input history and selector memory is easy to miss. Real-world `TBD` fields make the current file a research draft rather than a submission-ready manuscript. Fourteen references improve context, but recent flow-policy comparisons remain centered on A2A and Diffusion Policy.
- Technical failings that need to be addressed before the case is established: complete real-flight trials or remove physical-deployment language; state the single training seed explicitly; expand closed-loop scenarios; and add a short failure-mode analysis or trajectory visualization for cases where the selector slows progress.
- Assessment against Nature-style criteria: the work is original enough for a specialized robotics audience if the flow-specific control is added. Scientific importance and interdisciplinary reach are not demonstrated at a broad-journal level, though that is not required for ICRA. Technical reporting is careful. Readability is good after the method schematic, but the abstract can more clearly separate network behavior from selector behavior.
- Recommendation posture: readable and credible as a simulation paper, not yet ready for final submission because physical and multi-seed evidence is missing.

# Cross-review synthesis

- Consensus strengths: leakage-controlled evaluation, fair baseline retraining, honest proxy language, explicit intellectual-debt boundary, fast inference, and matched rollout-level continuity metrics.
- Consensus technical risks: absence of a same-backbone direct-regression control; single training seed; one closed-loop map with only five goals; ceiling success/collision outcomes; and incomplete real-flight evidence.
- Where emphasis differs across reviewers: Reviewer 1 prioritizes causal isolation and validation breadth; Reviewer 2 prioritizes novelty attribution across flow, cost learning, and selection; Reviewer 3 prioritizes deployment wording, selector memory, and submission completeness.
- Broad-interest / significance readout: the result is potentially meaningful within aerial robotics and structured robot policies, but the current evidence supports a bounded simulation contribution rather than a general claim that flow matching improves navigation.
- Most important issues to resolve before a strong case is established: train the matched direct-regression baseline; repeat main training across seeds; evaluate several map seeds and harder rollouts; complete physical trials; and keep the strongest continuity claim attached to the selector, not to deterministic flow alone.

# Risk / unsupported claims

- Unsupported if stated broadly: “flow matching is superior to direct regression.” No matched trained regression control exists.
- Weakly supported: cost-refined target search as the source of total-cost improvement. Its principal observed effect is on the offline safety proxy.
- Supported only in a narrow setting: 82% switch-rate reduction, measured for the stateful selector over five goals in one static map.
- Not assessable: real-world robustness, dynamic-obstacle performance, training-seed variance, and cross-platform generalization.
- Wording risk: “single-frame deployment” should refer to neural perception input; the optional selector carries the previous selected endpoint as one-step internal state.
