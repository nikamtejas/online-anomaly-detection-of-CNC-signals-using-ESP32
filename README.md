# Online anomaly detection of CNC signals using ESP32

Course project for Smart Sensor Systems at Leuphana University (WS 2026/27). It detects anomalies in CNC vibration data on an ESP32-S3, with models trained on normal recordings only.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Download the [Bosch dataset](https://github.com/boschresearch/CNC_Machining) and put it in `data/CNC_Machining/`, so that the machine folders end up in `data/CNC_Machining/data/`. Git ignores this folder.

## Scope

I restarted the project on 2026-11-12 with a smaller scope: one machine (M01) and one operation (OP01). M01 OP01 has 38 good and 2 bad recordings, and both bad ones are from 2019. The earlier work on all machines is saved under the git tag `v0-all-machines`.

## Progress

### Week 1: what the anomalies look like

1. `notebooks/01_data.ipynb` compares each bad OP01 recording with the good ones from the same period (2019-02: 5 good, 2019-08: 14 good). Both bad recordings have a higher RMS than every good one on all three axes. The difference is small on X and large on Y and Z: Z is 224 to 242 counts in the bad files and 159 to 193 in the good ones.
2. The extra energy sits around the 250 Hz spindle speed (paper, Table 1). The 200 to 400 Hz band is 5.1 to 8.7 dB higher in both bad files on every axis, and on Z the strongest peak moves to 246 to 248 Hz. So the machine shakes harder with every spindle turn. That fits an off-centre or unbalanced tool, but the data doesn't say what went wrong, so this is my reading and not a fact.
3. The bad files are also shorter (19.5 and 23.5 s, against 27.6 to 32.3 s), because the authors cut their start and end (paper p. 135). No model may use the length. Y below 50 Hz is about 7.5 dB lower in the bad files, possibly because the start and end spikes of a run are missing. I haven't tested that yet.

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
