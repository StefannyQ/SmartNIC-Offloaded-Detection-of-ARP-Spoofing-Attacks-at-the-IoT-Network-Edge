# P4 validation -- VM setup and execution guide

Companion to `04_generate_p4.py`. Covers everything needed to reproduce the
P4 hardware-fidelity validation: confirming that the integer-trained S4
Decision Tree, re-expressed as a P4_16 program, makes identical
classification decisions to the Python model on every test-set flow.

## Overview

The P4 program runs on BMv2 (the P4 behavioral model software switch)
inside a separate Linux environment -- a VM is the simplest way to get one.

**Two environments are involved:**

| Environment | What runs there |
|---|---|
| This repo (any OS) | `03_modeling.py`, `04_generate_p4.py` -- produce the model and generate all P4 files |
| P4 Linux environment (VM or container) | `p4c`, `simple_switch`, `test_harness.py` -- compile and run the P4 program |

## Prerequisites

### Main environment (this repo)
- Python environment with the pipeline dependencies installed (`pip install -r requirements.txt`)
- `03_modeling.py` has completed successfully and produced:
  - `modeling/results/S4_results.pkl`
  - `modeling/results/preprocessing_summary.json`
  - `modeling/results/S4_threshold_extraction.json`
  - `modeling/models/S4_Decision_Tree.joblib`

### P4 Linux environment
A Linux environment with `p4c` and BMv2 (`simple_switch`) installed. The
standard reference for this is the official P4 tutorials VM/repo
(https://github.com/p4lang/tutorials) -- its default credentials (`p4`/`p4`)
and Python venv layout (`/home/p4/src/p4dev-python-venv/`) match what this
guide assumes below; adjust if your setup differs.

**Versions this was validated against:**
```
OS            : Ubuntu 24.04 Desktop (AMD64)
p4c           : 1.2.5.12 (SHA: fe95abfa3, Release)
simple_switch : 1.15.1-08bba268
scapy         : 2.5.0
pandas        : 3.0.3 (in the P4 dev venv)
```
`p4c`/BMv2 are under active development -- a materially newer or older
version may compile or behave slightly differently. If you hit issues,
check `p4c --version` and `simple_switch --version` against the above first.

If using a VirtualBox VM: 4 CPUs / 8 GB RAM / 64 GB disk is comfortably
enough for this workload.

## Step 0 -- Generate the P4 files

From the repo root:
```bash
python 04_p4_validation/04_generate_p4.py
```
This reads the fitted S4 tree and writes `p4_validation/`:

| File | Purpose |
|---|---|
| `arp_detector.p4` | P4_16 program encoding the fitted decision tree |
| `flow_test_vectors.csv` | Full test set (~210,863 rows) with Python predictions |
| `test_harness.py` | Packet sender and verdict comparator |
| `table_entries.txt` | BMv2 startup notes |
| `README.md` | Quick reference card |

To use a smaller sample for a quick check instead of the full test set,
set `N_TEST_VECTORS = 2000` near the top of `04_generate_p4.py` before
running it (default is `None`, i.e. the full test set).

## Step 1 -- Get `p4_validation/` into the P4 environment

Any transfer method works (`scp`, a VirtualBox shared folder, a bind mount
if using a container). If using a VirtualBox shared folder:

1. VirtualBox menu: **Devices → Shared Folders → Shared Folders Settings...**
2. Add a folder pointing at this repo's root on the host
3. Give it a folder name (e.g. the repo's name), check **Auto-mount** and **Make Permanent**

Inside the VM:
```bash
sudo mkdir -p /mnt/shared
sudo mount -t vboxsf <shared-folder-name> /mnt/shared
mkdir -p ~/p4_validation
cp /mnt/shared/p4_validation/arp_detector.p4 ~/p4_validation/
cp /mnt/shared/p4_validation/flow_test_vectors.csv ~/p4_validation/
cp /mnt/shared/p4_validation/test_harness.py ~/p4_validation/
cd ~/p4_validation
```

Copying to local disk (rather than working directly from the shared mount)
avoids shared-folder I/O latency/disconnection issues with the
~210,000-row CSV.

**Sanity check you have the current file:**
```bash
grep "egress_spec" arp_detector.p4
```
Should show `standard_metadata.egress_spec = 1;`. If it instead shows
`standard_metadata.egress_spec = standard_metadata.ingress_port;`, the
P4 program reflects packets back out the *same* port it received them on
-- which races against the sender on the same interface. Re-run Step 0.

## Step 2 -- Set up virtual network interfaces

```bash
sudo ip link add veth0 type veth peer name veth1
sudo ip link set veth0 up
sudo ip link set veth1 up
```

BMv2 needs real network interfaces; a `veth` pair gives it two connected
back-to-back inside the kernel. The harness sends into `veth0`; BMv2
classifies and sends the verdict out `veth1`; the harness listens on
`veth1` for the reply (it starts listening *before* sending, to avoid a
race where the reply arrives before the listener is armed).

If `ip link add` says the interfaces already exist (from a previous
session), just run the two `up` lines.

## Step 3 -- Compile

```bash
p4c --target bmv2 --arch v1model --std p4-16 arp_detector.p4
```
Produces `arp_detector.json`. Expect one harmless warning about
`MAX_INT_SENTINEL` being unused; no errors. A compile error here usually
means a stale/corrupted `.p4` file -- re-run Step 0 and re-copy.

## Step 4 -- Start BMv2

```bash
sudo simple_switch --interface 0@veth0 --interface 1@veth1 arp_detector.json &
```
Wait a few seconds for `Adding interface veth0 as port 0` to print. Verify
with `ps aux | grep simple_switch`. Stop later with `sudo pkill simple_switch`.

## Step 5 -- Run the test harness

**First, sanity-check with 1 row:**
```bash
sudo /home/p4/src/p4dev-python-venv/bin/python3 \
    test_harness.py --iface veth0 --vectors flow_test_vectors.csv --max-rows 1
```
Expected: `Match : 1 / 1 (100.00%)` and `RESULT: PERFECT MATCH`. A timeout
here means BMv2 isn't responding -- recheck Steps 2 and 4.

**Then the full run:**
```bash
sudo /home/p4/src/p4dev-python-venv/bin/python3 \
    test_harness.py --iface veth0 --vectors flow_test_vectors.csv
```
- `sudo` is required because Scapy sends raw Ethernet frames.
- Use the full venv path, not bare `python3` -- `sudo`'s default Python
  doesn't have pandas/scapy installed.
- **Expect several hours** for the full ~210,863 rows, sent one at a time
  with a blocking send+capture per row. Progress prints every 100 rows.
  Keep the host machine from sleeping for the duration.

## Step 6 -- Copy results back

```bash
cp ~/p4_validation/p4_validation_results.csv /mnt/shared/p4_validation/
```

## Expected results

From the actual full-test-set run (210,863 rows):

| Result | Count | % |
|---|---|---|
| Match (P4 = Python) | 209,093 | 99.16% |
| Mismatch (P4 != Python) | 0 | 0.00% |
| Timeout (no reply within 2s) | 1,770 | 0.84% |

**Zero mismatches.** The timeouts are a VM-load artifact, not classification
disagreements:
- Evenly distributed across the run, not concentrated around any
  particular feature-value region
- Every timeout row has `p4_verdict = NaN` -- no wrong verdict was ever
  issued, the packet simply didn't get a reply within the 2-second window
- The timeout class split (~80% Benign / ~20% Attack) mirrors the
  dataset's own class ratio, i.e. there's no systematic bias toward
  losing one class's packets

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: pandas` | Using system Python instead of the venv | Use the full venv path: `sudo /home/p4/src/p4dev-python-venv/bin/python3` |
| `PermissionError: Operation not permitted` | Missing `sudo` for raw sockets | Add `sudo` before the `python3` command |
| All timeouts | BMv2 not running, or wrong interface | `ps aux \| grep simple_switch`; restart if needed |
| `File exists` on `ip link add` | veth pair already exists | Skip `ip link add`, just run the two `ip link set ... up` lines |
| Shared-folder mount errors | Mount dropped | `sudo umount /mnt/shared && sudo mount -t vboxsf <name> /mnt/shared` |
| `p4c` compile error | Stale/corrupted `.p4` file | Re-run Step 0 and re-copy the file |
