# Online anomaly detection of CNC signals using ESP32

Course project for Smart Sensor Systems at Leuphana University (WS 2026/27). It detects anomalies in CNC vibration data on an ESP32-S3, with models trained on normal recordings only.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Download the [Bosch dataset](https://github.com/boschresearch/CNC_Machining) and put it in `data/CNC_Machining/`, so that the machine folders end up in `data/CNC_Machining/data/`. Git ignores this folder.

## Scope

I restarted the project on 2026-11-12 with a smaller scope: machine M01 only, and only the 13 operations that have both good and bad recordings. OP00 and OP13 have no bad recordings, so they can't show whether the model detects anything, and I left them out before training anything. That leaves 413 good and 34 bad recordings. OP01 (38 good, 2 bad) is the worked example. The earlier work on all machines is saved under the git tag `v0-all-machines`.

## Progress

### Week 1: what the anomalies look like

1. `notebooks/01_data.ipynb` compares each bad OP01 recording with the good ones from the same period (2019-02: 5 good, 2019-08: 14 good). Both bad recordings have a higher RMS than every good one on all three axes. The difference is small on X and large on Y and Z: Z is 224 to 242 counts in the bad files and 159 to 193 in the good ones.
2. The extra energy sits around the 250 Hz spindle speed (paper, Table 1). The 200 to 400 Hz band is 5.1 to 8.7 dB higher in both bad files on every axis, and on Z the strongest peak moves to 246 to 248 Hz. So the machine shakes harder with every spindle turn. That fits an off-centre or unbalanced tool, but the data doesn't say what went wrong, so this is my reading and not a fact.
3. The bad files are also shorter (19.5 and 23.5 s, against 27.6 to 32.3 s), because the authors cut their start and end (paper p. 135). No model may use the length. Y below 50 Hz is about 7.5 dB lower in the bad files, possibly because the start and end spikes of a run are missing. I haven't tested that yet.
4. I ran the same comparison for every bad M01 recording (33 of 34; the OP10 one from 2021-08 has no good recording in its period to compare with). In 32 of 33, the energy within ±10% of the operation's own spindle speed is more than 6 dB above the good runs of the same group, and in 28 it's more than 10 dB. At first the 200 Hz operations (OP02, OP05, OP07) seemed to change between 100 and 200 Hz, but that was their spindle peak split by the band edge at 200 Hz.
5. The overall RMS rises much less than the spindle band. OP05 2019-08 gains 18.6 dB at its spindle speed and 0.1 dB in RMS, so an RMS threshold alone would miss it. The one exception is OP09 2019-08, which looks like a normal run on both measures, and I can't tell from the data why it's labelled bad. The model therefore needs a band around each operation's spindle speed, and so one model per operation. `results/figures/m01_bad_vs_good_heatmap.png` shows the differences per band and axis.

### Week 1: features per window

1. `notebooks/02_features.ipynb` cuts every recording into 1024-sample windows with a step of 256 (512 ms, one result every 128 ms) and computes 12 features per window: RMS, peak-to-peak, crest factor and the energy within ±10% of the spindle speed, each for X, Y and Z. A 250 Hz test sine gave RMS 0.7071, peak-to-peak 2 and crest factor 1.4142, as expected, and all 175,980 windows gave finite values.
2. I call a window idle when its largest axis RMS is below 30 counts. On a log scale the good windows show a small group at 10 to 25 counts, where the machine stands still, and a dip near 30. 0.52% of good windows are idle and no bad ones, because the bad files had their start and end cut. Dropping idle windows keeps the model from using that.
3. The anomaly lasts the whole run. In 30 of 33 bad recordings, more than 90% of windows are above the level that only 1% of normal windows of the same group reach on an axis (2.5% of good windows pass it, as three axes give three chances). So the board can see an anomaly within a few windows. OP09 2019-08 stays at 5% and won't be detected with this feature. The limits were fitted on the same good windows, so this describes the data and doesn't test a model.

### Week 1: first models

1. `notebooks/03_models.ipynb` trains two models per operation on good recordings only: an RMS threshold and the Mahalanobis distance with Ledoit-Wolf covariance, both on the 12 features with idle windows dropped. I split by recording. Within each operation, 60% of good recordings train the model, 20% set the threshold (99th percentile of their window scores) and 20% go to the test set with all 34 bad ones. A split by time would leave only 8 bad recordings to test. The alarm needs 3 of the last 5 windows above the threshold.
2. The flags aren't usable yet. Mahalanobis flags 8 of 34 bad test recordings and 2 of 84 good ones, the RMS baseline 11 of 34 and 1 of 84. Per window Mahalanobis does better (AUROC 0.791 against 0.685).
3. I found two causes. In OP10, normal windows score a median of 2.7 and the threshold is 9.3, while the bad windows sit at a steady 4.4 to 4.8, so the threshold guards against rare extreme normal windows and misses a steady, moderate excess. And normal moves between periods: OP08's good runs have an X spindle energy of 73.6 dB in 2019-02 and 84.3 dB in 2020-02, a shift as large as the anomaly. I have now seen the test results once, so the next test result won't be fully blind.
4. I fixed both causes, with all settings fixed before scoring the test set again. The centre of normal is now set per period from that period's good training recordings, as if the machine were recalibrated every 6 months. The alarm looks at the median score of the last 40 windows (about 5 s), which ignores short spikes and follows a steady rise. This needed a new split per operation and period, and OP10 2021-08 drops out because its period has no good recording to calibrate with.
5. Mahalanobis now flags 24 of 33 bad test recordings and 2 of 81 good ones, and window AUROC rose from 0.791 to 0.890. All 7 calibrated 2021 bad recordings are flagged. The RMS baseline flags 22 of 33 with 1 false flag. The price is a median delay of 5.5 s from the start of a bad file.
6. The misses come from the threshold, not the model. OP12's three bad recordings score 5.0 to 5.5 against 2.5 to 2.8 for normal runs, but one good validation recording scores 7.4 and lifts the threshold above them. In `results/figures/m01_score_over_time.png` the OP08 bad runs stay steadily above the good runs and below the threshold, and the good runs reach the threshold at their start and end.

## Data facts

"README" here means `data/CNC_Machining/README.md`, and "data" means the checks in the earlier analysis (tag `v0-all-machines`). Page numbers refer to the paper cited below.

| Quantity | Value | Unit | Source | Verified |
|---|---|---|---|---|
| Sampling rate | 2000 | Hz | README line 15, paper p. 132, data check below | yes |
| Signal unit | Not stated in any source | counts | README lines 15 and 37, paper, h5 files | **UNVERIFIED** |
| Axis order | Columns 0, 1, 2 are X, Y, Z | - | README line 37 | yes |
| Label meaning | good is a healthy process, bad is a faulty one | - | README line 28, paper p. 134 | yes |
| Machines | M01, M02, M03 | - | README line 26, paper p. 133, data | yes |
| Operations | 15 (OP00 to OP14), but M03 has no OP14 | - | README line 27, paper p. 133, data | yes |
| Recording period | October 2018 to August 2021. A filename's date marks the end of a 6-month interval | - | README line 15, paper p. 133 | yes |
| Good files | 1632 | files | Data | yes |
| Bad files | 70 | files | Data. The ratio matches the paper's 816:35 (p. 134) | yes |

The unit is the one fact I couldn't verify. The README and the paper don't name it, and the h5 files have no attributes. It is probably mg: the [CISS manual](https://www.manualslib.com/manual/1484706/Bosch-Ciss.html?page=8) lists 1 mg resolution and a ±16 g default range, the values are whole numbers, and Z sits near -1025. Until it is confirmed, everything stays in raw counts.

The files have no timestamps, so I checked the sampling rate against the spindle speeds in the paper's Table 1. Operations listed at 200 Hz peak between 199.7 and 200.0 Hz, and those listed at 250 Hz peak between 247.3 and 247.6 Hz. Median operation lengths come out 3% above the table.

These points still don't match, and I have no explanation for them:

- The paper says all 15 operations ran on all 3 machines, but M03 has no OP14.
- OP13 is listed at 75 Hz, but its strongest Z peak within 10% of that speed is at 67.6 Hz.
- The [UCI page](https://archive.ics.uci.edu/dataset/752/bosch+cnc+machining+dataset) lists 2700 instances, while this copy has 1702 files.

## Dataset and licence

The Bosch CNC Machining dataset is licensed under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) and isn't included in this repo.

> Tnani, M.-A.; Feil, M.; Diepold, K. Smart Data Collection System for Brownfield CNC Milling Machines: A New Benchmark Dataset for Data-Driven Machine Monitoring. Procedia CIRP 2022, 107, 131-136. https://doi.org/10.1016/j.procir.2022.04.022
