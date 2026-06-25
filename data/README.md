# Data

This folder is where the raw dataset goes. **Nothing in this folder is
committed to the repo** -- CICIoT2023's usage terms do not permit
redistributing the raw data.

## Download

- Source: CIC Research, University of New Brunswick — https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/
- Cite: Neto, E.C.P. et al., "CICIoT2023: A Real-Time Dataset and Benchmark
  for Large-Scale Attacks in IoT Environment," *Sensors*, 23(13):5941, 2023.

The full CICIoT2023 release has one folder per attack category (DDoS-*,
DoS-*, Mirai-*, Recon-*, MITM-ArpSpoofing, ...) plus benign traffic, as
CSVs already processed by CICFlowMeter (not raw pcaps). This pipeline only
needs two of them:

- `Benign_Final/` (benign IoT traffic flows, label = 0)
- `MITM-ArpSpoofing/` (ARP spoofing attack flows, label = 1)

## Where to put it

Copy those two folders, unmodified, into this `data/` folder so the layout
is:

```
data/
  Benign_Final/*.csv
  MITM-ArpSpoofing/*.csv
```

`01_data_cleaning/01_data_cleaning.py` globs every `*.csv` in each folder,
so extra or renamed files are fine as long as the two folder names match.
Once you run the pipeline, this same folder also receives the cleaned
checkpoint:

```
data/
  final_dataset.csv      <- produced by 01_data_cleaning.py
```

No path edits are needed anywhere -- every script resolves `data/`
relative to the repo root automatically.
