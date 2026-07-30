# LatticeFlow ICRA 2027 Paper Workspace

This directory contains the double-anonymous ICRA 2027 manuscript and its evidence trail. The paper is based on the implementation in `/home/hwm/CF_YOPO/YOPO_FlowMatching` and the two source papers in `/home/hwm/CF_YOPO/paper`.

## Central argument

LatticeFlow transports deterministic, physically meaningful YOPO motion-primitive anchors to cost-improved terminal states with a depth-and-state-conditioned flow. A frozen YOPO model provides an initial target proposal, while student exploration and differentiable trajectory costs can improve that proposal beyond the teacher. Local consistency regularization and continuity-aware lattice selection address frame-to-frame route switching without adding historical images or actions to the neural input.

## Intellectual-debt boundary

- From YOPO: spherical motion-primitive lattice, normalized terminal-state residuals, fifth-order polynomial reconstruction, shared fully convolutional prediction and privileged ESDF guidance.
- From A2A: the insight that flow-based robot policies need not begin from uninformed Gaussian noise and can instead use a structured source closer to the action manifold.
- Introduced here: canonical lattice anchors as the structured flow source for depth-based planning; teacher/student/noise candidate selection followed by differentiable cost refinement; depth-projection safety; local perturbation consistency; and continuity-aware lattice selection.

The implementation is not an A2A history-action policy because the YOPO dataset contains independent depth poses rather than temporal action sequences.

## Official ICRA 2027 constraints

- Double-anonymous review.
- Complete manuscript limit: 8 pages including references and acknowledgments.
- PDF in ICRA double-column format.
- Submission deadline: 15 September 2026, 11:59 PST.
- The official PaperCept `ieeeconf` class is included in `paper/`.

## Build

```bash
/home/hwm/miniconda3/bin/conda run -n paper \
  latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -cd /home/hwm/CF_YOPO/LatticeFlow_ICRA2027/paper/main.tex
```

The current manuscript uses explicit `TBD` macros for experimental values that have not yet been generated. These placeholders must be resolved before submission.

