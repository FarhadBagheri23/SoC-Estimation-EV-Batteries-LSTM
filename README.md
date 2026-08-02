# Anomaly Detection in EV Battery Charging — State of Charge (SoC)

LSTM-based SoC estimation and sensor-fault detection on the LG 18650HG2 cell.

Everything lives in one self-contained notebook, [`SOC_NOTEBOOK.ipynb`](SOC_NOTEBOOK.ipynb),
which runs end to end in Google Colab on a T4 GPU in roughly 45 minutes.

---

## The problem

A battery management system reports State of Charge, but it cannot *measure* it — no
sensor reads SoC directly. The BMS infers it, almost always by Coulomb counting:
integrating the measured current over time. Integration means any error in the current
sensor is **accumulated rather than averaged away**. A 20 mA offset on a 3 Ah cell is
invisible in any single sample and becomes several percent of SoC over an hour.

This is the central reliability problem in SoC estimation, and it needs no external
cause. Sensors drift, connectors loosen, capacity fades, and the reported SoC quietly
separates from the truth.

This work builds an independent second opinion. A recurrent network predicts SoC from
the raw sensor stream — voltage, current, temperature — using patterns learned across
the full automotive temperature range. When the BMS-reported SoC and the model-predicted
SoC disagree *persistently*, something in the measurement chain is wrong, and the
detector reports it before the error compounds.

## Data

The McMaster benchmark published by Kollmeyer et al. (2020): a 3 Ah LG 18650HG2 cell
cycled in a thermal chamber at six ambient temperatures from −20 °C to +40 °C, driven
through UDDS, LA92, US06, HWFET and mixed automotive profiles alongside characterisation
tests. The notebook pulls it from Kaggle, with a two-file public mirror as a fallback so
it still runs end to end if the primary host is unreachable.

Raw logs at ~10 Hz are reduced to 1 Hz by **block-averaging before decimating**, not by
keeping every tenth sample. Drive-cycle current is spiky, and because the SoC label is
derived by integrating current, the physically meaningful quantity at 1 Hz is the mean
current over each one-second bin.

Each training sample is a 100-step history of 8 features (the three measured channels,
power, and 60 s / 300 s rolling means of voltage and current); the label is SoC at the
final step.

## What makes the numbers trustworthy

Two rules decide whether reported accuracy on this problem is real:

- **Split by whole drive cycle, never by window.** Consecutive windows overlap by 99 of
  their 100 samples, so a random window split puts near-duplicates in both train and
  test. That measures memorisation, not generalisation.
- **Stratify by temperature.** Cycles are sorted by mean temperature and dealt
  round-robin into the three partitions, so train, validation and test each span the
  full −20 °C to +40 °C range.

The test partition is read exactly once, after the final architecture is chosen.

## Estimation results

Six architectures under identical data, splits, parameter budgets and training
procedure. Each is trained three times from different seeds, because the spread across
repetitions can exceed the difference between two designs (Vidal et al., 2020).

Test MAE in percentage points of SoC — an MAE of 0.8 means the estimate is wrong by
0.8 % of full charge on average:

| Model | MAE | ± sd | RMSE | R² |
|---|---|---|---|---|
| LSTM multi-task | **0.814** | 0.026 | 1.325 | 0.998 |
| Bi-LSTM + attention | **0.821** | 0.015 | 1.385 | 0.998 |
| Proposed hybrid | 0.850 | 0.005 | 1.357 | 0.998 |
| CNN-LSTM | 0.959 | 0.014 | 1.694 | 0.997 |
| FNN | 1.136 | 0.010 | 1.945 | 0.996 |
| Ridge baseline | 4.591 | — | 7.149 | 0.945 |

**The top two are tied.** The gap is 0.007 pp against a pooled seed spread of 0.026 pp,
so no winner is claimed. The margin of every deep model over the ridge baseline is far
too large to be initialisation luck.

Error is not uniform across conditions — LSTM-MT scores 1.00 pp in the cold (< −5 °C)
against 0.40 pp when warm, a 2.5× ratio. That result drives the detector design: if
estimation error depends on operating condition, a single fixed alarm threshold cannot
be optimal, so the residual is normalised per temperature bin.

## Fault detection

Every 60 s the BMS reports the SoC increment ΔSoC. Two independent estimates of it
exist — the reported one, from Coulomb counting, and the model's, recomputed from raw
voltage, current and temperature. The detector monitors the residual
`DM = |ΔSoC_rep − ΔSoC_pred|`, normalised by the healthy 95th-percentile scale of its
temperature bin, and flags a ten-report window when the mean normalised residual clears
a threshold.

That threshold is calibrated on **healthy validation streams only**, at a stated alarm
budget. No test data and no faulty data of any kind enter the calibration — which is
what makes the procedure deployable, since a BMS can calibrate against its own healthy
operating history without ever having observed a fault.

Three fault modes are injected into contiguous thirty-minute blocks: **bias** (constant
offset — calibration error), **drift** (arithmetic progression — sensor ageing, capacity
fade) and **noise** (random draws — loose connector, ADC fault).

AUROC against fault magnitude, five injection seeds per point:

| magnitude (pp) | 0.3 | 0.5 | 0.9 | 1.5 | 2.0 | 3.0 | 4.0 | 6.0 |
|---|---|---|---|---|---|---|---|---|
| AUROC | 79.9 | 83.9 | 89.2 | 93.3 | 95.2 | 97.3 | 98.3 | 99.3 |

For scale, the true mean \|ΔSoC\| is about 0.9 pp per report, so a 0.9 pp fault is the
same size as the signal itself.

**The ordering of difficulty is not the intuitive one.** Recall at a 2 % alarm budget:

| fault | 0.9 pp | 2.0 pp | 6.0 pp |
|---|---|---|---|
| bias | 19.9 | 69.4 | 100.0 |
| drift | 18.7 | 33.5 | 76.8 |
| noise | 45.6 | 61.3 | 97.9 |

Small noise faults are the *easiest* to catch: noise replaces the reported increment
with a draw near zero while the cell is genuinely moving at ~0.9 pp per minute, so the
telemetry has effectively flatlined. Bias is hardest at low magnitude — it adds to the
true value, so the stream keeps tracking real dynamics — but then transitions sharply.
Drift is the genuinely hard mode: it starts at zero error by construction and only
becomes visible late in the block.

Beyond alarming, the detector diagnoses which mode occurred, from the coefficient of
variation of the residual (flat plateau → bias) and the R² of a straight-line fit
(clean ramp → drift). No training, two cut-points: 41 % accuracy at 0.9 pp, 80 % at
3.0 pp, 99 % at 6.0 pp, against 33 % chance.

## Limitations

Stated plainly, because they bound how these figures should be quoted:

- **The evidence base for detection is thin.** 13 test cycles, 1,462 reports, 24.4 hours,
  1,168 scored windows. The injection seeds resample fault placement, not the underlying
  cycles, so the error bars understate true uncertainty.
- **SoC is defined per file** over the amp-hour counter, and retained spans vary 1.7×.
  That variation tracks temperature (r = +0.84) — usable capacity genuinely shrinks in
  the cold — so the definition is valid, but it is *percentage of usable capacity at
  that temperature* and must be quoted that way.
- **Accuracy is inflated by class balance.** 29.8 % of windows carry faults, so a
  detector that never alarms already scores 70.2 %. Quote AUROC and recall at a stated
  false-alarm rate, with the lift over that trivial baseline.
- **Faults are injected, not observed.** The three modes are the standard taxonomy, but
  no real failed sensor appears in this data.
- **One cell chemistry, one dataset.** Nothing here demonstrates transfer to another
  cell or pack.

## Running it

Open the notebook in Colab, select a GPU runtime, and run all cells. Dependencies are
installed in-notebook; the dataset downloads automatically. On CPU it will complete but
the training section takes hours rather than minutes.

## Key references

Kollmeyer, P., Vidal, C., Naguib, M., & Skells, M. (2020). LG 18650HG2 Li-ion battery
data and example deep neural network xEV SOC estimator script. *Mendeley Data, V3*.

Chemali, E., Kollmeyer, P. J., Preindl, M., Ahmed, R., & Emadi, A. (2018). Long
short-term memory networks for accurate state-of-charge estimation of Li-ion batteries.
*IEEE Transactions on Industrial Electronics, 65*(8), 6730–6739.

Vidal, C., Malysz, P., Kollmeyer, P., & Emadi, A. (2020). Machine learning applied to
electrified vehicle battery state of charge and state of health estimation:
State-of-the-art. *IEEE Access, 8*, 52796–52814.

Mitikiri, S. B., Tiwari, Y., Srinivas, V. L., & Pal, M. (2024). Regression based anomaly
detection in electric vehicle state of charge fluctuations through analysis of EVCI data.
*arXiv preprint arXiv:2401.01580*.

The full reference list is in the notebook's closing section.
