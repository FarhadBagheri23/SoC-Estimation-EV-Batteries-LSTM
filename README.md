<div align="center">

<img src="assets/sharif-logo.jpg" width="115" alt="Sharif University of Technology">

<h1>State-of-Charge Estimation for EV Lithium-Ion Batteries</h1>

<b>Sharif University of Technology — International Campus, Kish Island</b><br>
B.Sc. Thesis Project · Department of Computer Engineering · 2026<br>
<b>Farhad Bagheri Taheri</b> &nbsp;·&nbsp; Supervisor: <b>Dr. Amin Foshati</b>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FarhadBagheri23/SoC-Estimation-EV-Batteries-LSTM/blob/main/SOC_NOTEBOOK.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

A controlled comparison of deep learning architectures for **State of Charge (SoC)**
estimation on the LG 18650HG2 cell, across the full automotive temperature range of
−20 °C to +40 °C.

Everything lives in one self-contained notebook, [`SOC_NOTEBOOK.ipynb`](SOC_NOTEBOOK.ipynb),
which runs end to end in Google Colab on a T4 GPU. The full comparison — five deep
architectures across three seeds plus a ridge baseline, sixteen fitted models — takes
about 37 minutes of training.

---

## The problem

A battery management system reports State of Charge, but it cannot *measure* it — no
sensor reads SoC directly. It is inferred, almost always by Coulomb counting:
integrating the measured current over time. Integration means any error in the current
sensor is **accumulated rather than averaged away**. A 20 mA offset on a 3 Ah cell is
invisible in any single sample and grows to roughly two-thirds of a percentage point of
SoC for every hour of operation.

Estimating SoC from the measured signals directly avoids that accumulation, and two
properties of the cell are what make it hard. The relationship between terminal voltage
and charge state is weak across the middle of the usable window, so a single reading
carries little information. And that relationship shifts substantially with temperature
— the validity audit here measures a 28 % contraction in usable capacity between the
warm and cold ends of this benchmark. The estimator must therefore read a *window* of
recent history and be conditioned on temperature, which is what every architecture
compared here does.

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

Any file spanning less than 1.5 Ah of the 3 Ah cell is rejected, since the per-file SoC
normalisation is only meaningful over a substantial portion of the charge range. That
leaves **64 cycles and 443,390 one-second samples**.

Each training sample is a 100-step history of 8 features — the three measured channels,
power, and 60 s / 300 s rolling means of voltage and current — labelled by the SoC at
the final step. Cumulative amp-hours is deliberately excluded: it is the quantity the
label is derived from, and feeding it in would leak the target.

## What makes the numbers trustworthy

Two rules decide whether reported accuracy on this problem is real:

- **Split by whole drive cycle, never by window.** Consecutive windows overlap by 99 of
  their 100 samples, so a random window split puts near-duplicates in both train and
  test. That measures memorisation, not generalisation.
- **Stratify by temperature.** Cycles are sorted by mean temperature and dealt
  round-robin into the three partitions, so train, validation and test each span the
  full −20 °C to +40 °C range.

Section 5.3 checks this rather than asserting it: that the three index sets are disjoint
and complete, that no source file was loaded twice, and that no two cycles in different
splits are near-duplicates of the same physical test. Standardisation statistics are
fitted on the training partition alone, and the test partition is read exactly once,
after the final architecture is chosen on validation error.

## Results

Six estimators under identical data, splits, parameter budgets and training procedure.
Each deep model is trained three times from different seeds, because the spread across
repetitions can exceed the difference between two designs (Vidal et al., 2020).

Test MAE in percentage points of SoC — an MAE of 0.8 means the estimate is wrong by
0.8 % of full charge on average:

| Model | MAE | ± sd | RMSE | R² | Params |
|---|---|---|---|---|---|
| LSTM multi-task | **0.814** | 0.026 | 1.325 | 0.998 | 79,106 |
| Bi-LSTM + attention | **0.821** | 0.015 | 1.385 | 0.998 | 158,082 |
| Proposed hybrid | 0.850 | 0.005 | 1.357 | 0.998 | 90,115 |
| CNN-LSTM | 0.959 | 0.014 | 1.694 | 0.997 | 122,625 |
| FNN | 1.136 | 0.010 | 1.945 | 0.996 | 25,985 |
| Ridge baseline | 4.591 | — | 7.149 | 0.945 | — |

**The top two are tied.** The gap is 0.007 pp against a pooled seed spread of 0.026 pp,
a ratio of 0.27, so no winner is claimed. The same experiment reported from seed 42
alone would have shown Bi-LSTM + attention ahead by 0.047 pp; from seed 44 alone, the
multi-task LSTM ahead by 0.048 pp. Two opposite conclusions of near-identical magnitude,
from the same code and the same data, differing only in which initialisation was
reported — which is exactly why every architecture here is trained more than once.

The margin of every deep model over the ridge baseline is far too large to be
initialisation luck, and is stated without hedging.

**Recurrence matters.** The feed-forward network, which sees only the final timestep
plus the rolling summaries, trails every sequence model by at least 0.18 pp. The rolling
means are not a substitute for temporal structure.

### The proposed hybrid, and why it is reported as a negative result

The hybrid combines four ideas that each help in isolation: a convolutional front end
(Song et al., 2019), bidirectional recurrence (Terala et al., 2022), attention pooling
(Mamo & Wang, 2020; Li et al., 2026) and an auxiliary voltage-reconstruction head (Ma et
al., 2025). Stacking them did **not** produce an additive improvement — it ranks third.

Two properties still distinguish it, both read off the data rather than argued around
it. It is by a wide margin the **most reproducible** architecture tested: a seed spread
of ±0.005 pp against ±0.026 pp for the leader, with its three runs spanning less than a
single reseeding moves the leader. And it is the **second most accurate below −5 °C**
(1.056 pp against 1.084 pp for Bi-LSTM + attention), the regime where data-driven
estimators most often fail. On a dataset of this size, extra architectural machinery
appears to give diminishing and then negative returns.

### Error is not uniform

MAE by temperature bin, percentage points:

| Bin | FNN | CNN-LSTM | Bi-LSTM+Attn | LSTM-MT | Hybrid | Ridge |
|---|---|---|---|---|---|---|
| cold (< −5 °C) | 1.822 | 1.465 | 1.084 | **1.002** | 1.056 | 8.856 |
| cool (−5…15 °C) | 1.023 | 0.944 | 0.942 | **0.868** | 0.942 | 2.583 |
| warm (> 15 °C) | 0.499 | 0.441 | **0.363** | 0.403 | 0.409 | 2.826 |

The selected estimator is 2.49× less accurate below −5 °C than above 15 °C, and no
single architecture leads in every regime. The aggregate figure describes no operating
condition in particular.

## Limitations

Stated plainly, because they bound how these figures should be quoted:

- **SoC is defined per file** over the amp-hour counter, and retained spans vary 1.7×.
  That variation tracks temperature (r = +0.84 over 64 cycles) — usable capacity
  genuinely shrinks in the cold — so the definition is valid, but it is *percentage of
  usable capacity at that temperature* and must be quoted that way.
- **No hyperparameter search.** Every setting is fixed in advance so that no model
  benefits from more tuning effort than another. The absolute errors are therefore not
  the best achievable on this cell, and much of the gap to Benallal et al. (2025), who
  tuned with Hyperband on the same data and report 0.43 %, plausibly sits here.
- **Three seeds.** Enough to show the leading gap is not robust, not enough for a formal
  significance test.
- **Cold, low-SoC operation is the weakest regime**, with error spiking above 4 pp near
  end of discharge in the coldest cycles.
- **Discharge cycles dominate.** The screening rule retains dynamic drive cycles and
  full-range discharges, so charge-phase behaviour is under-represented.
- **One cell, laboratory data.** Nothing here demonstrates transfer to another cell or
  to pack level, where cell-to-cell variation, balancing currents and BMS filtering all
  enter.

## Running it

Open the notebook in Colab, select a GPU runtime, and run all cells. Colab already ships
every dependency, so there is nothing to install, and the dataset downloads
automatically. On CPU the notebook still completes, but the training section takes hours
rather than minutes.

To run it locally instead, `pip install -r requirements.txt`. The results above were
produced on torch 2.11.0+cu128 with a Tesla T4.

## Key references

Kollmeyer, P., Vidal, C., Naguib, M., & Skells, M. (2020). LG 18650HG2 Li-ion battery
data and example deep neural network xEV SOC estimator script. *Mendeley Data, V3*.

Chemali, E., Kollmeyer, P. J., Preindl, M., Ahmed, R., & Emadi, A. (2018). Long
short-term memory networks for accurate state-of-charge estimation of Li-ion batteries.
*IEEE Transactions on Industrial Electronics, 65*(8), 6730–6739.

Vidal, C., Malysz, P., Kollmeyer, P., & Emadi, A. (2020). Machine learning applied to
electrified vehicle battery state of charge and state of health estimation:
State-of-the-art. *IEEE Access, 8*, 52796–52814.

Song, X., Yang, F., Wang, D., & Tsui, K.-L. (2019). Combined CNN-LSTM network for
state-of-charge estimation of lithium-ion batteries. *IEEE Access, 7*, 88894–88902.

Ma, L., Li, Y., Zhang, T., Tian, J., Guo, Q., Guo, S., Hu, C., & Chung, C. Y. (2025).
Trustworthy battery state of charge estimation enabled by multi-task deep learning.
*Energy, 326*, 136264.

The full reference list is in the notebook's closing section.

## License

MIT — see [LICENSE](LICENSE). The LG 18650HG2 dataset is the property of its authors
(Kollmeyer et al., 2020) and carries its own terms; this license covers only the code
and analysis in this repository.
