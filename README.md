# Online anomaly detection of CNC signals using ESP32

Course project for Smart Sensor Systems at Leuphana University (WS 2026/27). It detects anomalies in CNC vibration data on an ESP32-S3, with models trained on normal recordings only.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Download the [Bosch dataset](https://github.com/boschresearch/CNC_Machining) and put it in `data/CNC_Machining/`, so that the machine folders end up in `data/CNC_Machining/data/`. Git ignores this folder.

## Run

```bash
cd notebooks
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```

The firmware check needs MicroPython (`brew install micropython`) and runs from the repo root:

```bash
.venv/bin/python -m sim.run_sim
```

## Progress

### Week 1: setup and data

1. I put the Bosch dataset in `data/CNC_Machining/` and told git to ignore it. It has 1702 recordings (24.81 hours), 1632 good and 70 bad. Each file holds one array, `vibration_data`, with one column per axis.
2. `src/features.py` loads a recording and cuts it into windows of 1024 samples, starting a new window every 256 samples. Each window gives 11 values per axis: RMS, peak-to-peak, crest factor and 8 log FFT band energies. That makes 33 features. A test sine wave gave the expected RMS (0.7071), peak-to-peak (2.000) and crest factor (1.414), and all 692,452 windows in the dataset gave finite values.
3. I compared good and bad spectra in `notebooks/analysis.ipynb`, using only the 46 groups that have both labels for the same machine, operation and period. Bad recordings carry more energy at 178 to 316 Hz: 9.9 dB on X, 8.6 dB on Y and 10.5 dB on Z. No other band differs by more than 2.2 dB.
4. I wrote the anomaly definition from the dataset README and the paper. A recording is an anomaly if it is in a `bad/` folder. The paper names tool misalignment, chip clamping, chips in the chuck and tool breakage as causes (p. 134).
5. I checked how the bad files were prepared, because the authors cut off their start and end during labelling (p. 135). They are a median of 0.82 times as long as good files in the same group and rarely contain idle windows. Scoring recordings only by "less idle means bad" already reaches AUROC 0.926, so the models need testing with idle windows removed.
6. I checked the dataset facts against their sources. The results are in the table below.

The ESP32 smoke test is still open, because the board isn't connected yet. I removed the unused upstream files from the dataset folder, so it now holds only the data, LICENSE and README.

### Week 2: models (first run)

1. I split the data by year, so training comes before testing. Good 2019 recordings train the models (809), good 2020 sets the thresholds (371), and all of 2021 is the test set (452 good, 17 bad). The 53 bad recordings from 2019 stay unused for now.
2. `src/train.py` fits the RMS baseline, Mahalanobis distance and PCA reconstruction error. `src/evaluate.py` adds the accuracy rules one at a time and logs each result to `results/metrics.csv`.
3. None of the rules gave a clear gain. On recordings, Mahalanobis AUROC was 0.899 with none of them and 0.893 with all four. The 17 bad recordings make small differences hard to trust, since one recording moves recall by 5.9 points.

| Model, all rules | AUROC (recording) | PR-AUC (recording) | Recall at 1% FPR (recording) | AUROC (window) | False alarms per hour |
|---|---|---|---|---|---|
| RMS | 0.847 | 0.307 | 0.000 | 0.787 | 176 |
| Mahalanobis | 0.893 | 0.276 | 0.000 | 0.890 | 581 |
| PCA | 0.860 | 0.284 | 0.059 | 0.881 | 594 |

The models rank recordings fairly well but aren't usable as alarms yet. At 1% false positives they catch almost no bad recordings, and 2.07% of normal test windows pass the Mahalanobis threshold, twice the 1% target. False alarms are counted before alarm smoothing, which comes in Week 3.

### Week 2: why recall is zero

1. Six normal recordings take ranks 1 to 6 under Mahalanobis, and all six are M01 OP13 from 2021-02. With 452 normal test recordings, 1% FPR allows 4 false positives, so these six alone push recall to 0.
2. They differ from the other nine good recordings in that group. Their strongest Y peak sits at 947 to 992 Hz instead of 338 Hz, and their Y RMS is 595 to 722 counts instead of 170 to 184. In the 562 to 1000 Hz band they are 8.4 to 12.8 dB above the M01 OP13 training recordings.
3. I can't say whether they are mislabelled. The paper admits some labels are wrong (p. 134), but nothing here proves it for these files, and OP13 has no bad recordings on any machine to compare with. The labels stay as published, so the table above remains the official result.
4. Leaving the six out moves Mahalanobis PR-AUC from 0.276 to 0.561 and recall at 1% FPR from 0 to 0.471. RMS and PCA move the same way. I picked the six after looking at the test set, so this explains the result and doesn't improve it.
5. False alarms aren't evenly spread either. OP13 has 10.51% of its normal windows above the threshold, and no other operation passes 4.07%. The rate rises from 1.67% in 2021-02 to 3.19% in 2021-08. The last tenth of a recording holds 25.3% of them and the first tenth 15.3%, where an even spread would give 10%.

### Week 3: simulation

1. `firmware/features.py` computes the same 33 features in MicroPython. `sim/run_sim.py` feeds recorded samples through a ring buffer in the MicroPython Unix port and compares every window with `src/features.py`. I tested 362 windows: one good and one bad 2021 recording per machine, plus a flat signal. The largest differences were 0.00095 counts for RMS, 0 for peak-to-peak, 0.00016 for crest factor and 0.0000076 for log band energy.
2. This doesn't settle everything about the chip. `emlearn_fft` has no build for my arm64 Mac, so the simulation uses a Python copy of its FFT. The Unix port also computes in 64-bit floats, while the ESP32 uses 32-bit, and laptop speed says nothing about the 128 ms budget. All three need the board in Week 4.
3. `notebooks/04_simulation.ipynb` replays the 2021 test recordings (452 good, 17 bad) one after another, the way the board will get them. Only the periods have a known time order, so inside a period I sort by machine, operation and file number. I added rule 5: the alarm goes on only when 3 of the last 5 windows are above the threshold, and the count starts again with every recording.
4. Rule 5 cuts false alarms from 581 to 438 per hour and raises window AUROC from 0.890 to 0.914. For whole recordings it doesn't help. AUROC falls from 0.893 to 0.878, recall at 1% FPR stays at 0, and 45% of normal recordings still get at least one alarm. 12 of the 17 bad recordings raise an alarm, a median 1.92 s after the file starts. The authors cut off the start of the bad files, so this isn't the delay from the start of the fault. `results/figures/score_over_time.png` shows one stream (M01, 2021-08).

## Data facts

"README" here means `data/CNC_Machining/README.md`, and "data" means the output of `notebooks/analysis.ipynb`. Page numbers refer to the paper cited below.

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
