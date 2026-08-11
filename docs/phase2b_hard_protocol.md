# SRRD Phase 2B-hard — End-to-End Inductive-Bias Falsification

Phase 2 ended at **C: history dependence survives, SRRD decomposition does not**. The decisive limitation was that the strongest competitors were fixed-feature/fixed-core screening models rather than end-to-end GRU/LSTM, learned predictive-state models, and an actual action-selecting History-MPC.

Phase 2B-hard changes only that unresolved point. It does **not** reopen the already-frozen scientific wording or lower the Phase 2 thresholds.

## Question

After observable state matching and strict train-only nested tuning, does an explicit SRRD fast/slow decomposition earn OOD predictive value that capacity-matched end-to-end alternatives cannot reproduce?

The primary estimand is

`R_OOD = NLL(SRRD) / min NLL(nominal end-to-end baselines)`.

Support requires the 95% bootstrap upper CI to be `<= 0.90`. A separate 4x recurrent capacity attack uses the same `<= 0.90` rule. History shuffle, frozen slow-update, and `psi_update` remain mechanism requirements, not substitutes for OOD superiority.

## Hardening relative to Phase 2

- Real end-to-end GRU and LSTM baselines.
- Learned predictive-state bottleneck trained on future outcomes.
- Action-conditioned multi-step History-MPC with counterfactual action selection as an implementation-validity audit.
- Explicit end-to-end SRRD bi-level model with slow history state, fast observable state, and a learned slow-update gate.
- Trainable parameters matched around 2,500 within 10% for all nominal families.
- The identical six-trial optimization grid, epoch ceiling, early stopping, and residual calibration protocol for every family.
- OOD `C2=1.2` is forbidden from tuning and calibration.
- 4x recurrent capacity attack is analyzed separately from nominal parameter matching.
- Independent verifier recomputes ratios, parameter fairness, freeze hash, gates, and final classification.

## Run modes

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_phase2b_hard.py --mode smoke
python scripts/verify_phase2b_hard.py --mode smoke --results results/phase2b_hard_smoke
```

Development uses reduced seeds/epochs only to debug implementation. It may not change frozen thresholds. The full confirmatory command is deliberately separate:

```bash
python run_phase2b_hard.py --mode confirmatory --output results/phase2b_hard_confirmatory
python scripts/verify_phase2b_hard.py --mode confirmatory --results results/phase2b_hard_confirmatory
```

Confirmatory execution requires the SHA-256 in `preregistration/phase2b_hard.freeze.json` to match exactly. Any post-freeze scientific change requires a new study ID.

## Interpretation

- **A**: explicit SRRD decomposition survives end-to-end and 4x capacity attacks under valid controls.
- **B**: history/reconstruction effects remain, but SRRD is not uniquely required.
- **C**: SRRD wins nominally but loses to the 4x recurrent capacity attack; decomposition is capacity-sensitive rather than necessary.
- **D**: the SRRD mechanism itself fails its order/update/reconstruction controls.
- **E**: state matching, capacity fairness, or actual History-MPC competence makes the comparison uninterpretable.
- **F**: implementation/data/tuning isolation fails.

Until a frozen confirmatory run changes it, the scientific wording remains: **history matters; SRRD is not uniquely required**.
