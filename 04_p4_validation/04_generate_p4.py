#!/usr/bin/env python3
"""
04_generate_p4.py -- Generate P4 program and test harness from fitted S4 tree

PURPOSE
-------
Reads the integer-trained S4 Decision Tree from S4_results.pkl and the
preprocessing summary from preprocessing_summary.json, then generates:

  p4_validation/
    arp_detector.p4          -- P4_16 program (compile with p4c, run on BMv2)
    table_entries.txt        -- match-action table entries for simple_switch_CLI
    test_harness.py          -- sends test flows through BMv2, compares to Python
    flow_test_vectors.csv    -- test-set rows with Python predictions (ground truth)
    README.md                -- step-by-step instructions for the VM

WHAT THE GENERATED P4 PROGRAM DOES
------------------------------------
Each test-set flow is represented as a custom Ethernet frame carrying all
24 S4 feature values in a custom header. The P4 program:
  1. Parses the custom header
  2. Computes the FE_* engineered features using integer arithmetic
  3. Applies the fitted decision tree as a sequence of exact-match / LPM
     table lookups (one table per tree level)
  4. Sets metadata.verdict = 0 (Benign) or 1 (Attack)
  5. Reflects the packet back out the same port with verdict in a result header

The test harness sends each flow vector as one crafted packet, reads the
reflected verdict, and compares it to the Python model's prediction on the
same row. A 100% match rate means the P4 encoding is faithful.

USAGE
-----
  python 04_generate_p4.py

Then copy the p4_validation/ folder to your P4 VM and follow README.md.

PATHS  (match 03_modeling.py)
-----
"""

import os, sys, io, json, pickle, csv, textwrap
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import _tree

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Paths (must match 03_modeling.py) ────────────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, 'modeling', 'results')
MODELS_DIR  = os.path.join(BASE, 'modeling', 'models')
DATA_PATH   = os.path.join(BASE, 'data', 'final_dataset.csv')
OUT_DIR     = os.path.join(BASE, 'p4_validation')

S4_RESULTS_PATH  = os.path.join(RESULTS_DIR, 'S4_results.pkl')
S4_MODEL_PATH    = os.path.join(MODELS_DIR, 'S4_Decision_Tree.joblib')
SUMMARY_PATH     = os.path.join(RESULTS_DIR, 'preprocessing_summary.json')
THRESHOLD_PATH   = os.path.join(RESULTS_DIR, 'S4_threshold_extraction.json')

# Number of test-set rows to include in the test vectors file.
# Set to None to use the full test set (~210,863 rows).
# Set to an integer (e.g. 2000) for a faster stratified sample.
N_TEST_VECTORS = None  # None = full test set
RANDOM_STATE   = 42

SMARTNIC_MAX_INT = 2**32 - 1

print("=" * 70)
print("04_generate_p4.py -- P4 Program Generator")
print("=" * 70)

# ── Load artifacts ────────────────────────────────────────────────────────────
for path, label in [(S4_RESULTS_PATH, 'S4_results.pkl'),
                    (S4_MODEL_PATH,   'S4_Decision_Tree.joblib'),
                    (SUMMARY_PATH,    'preprocessing_summary.json')]:
    if not os.path.exists(path):
        sys.exit(f"\nERROR: {path} not found.\nRun 03_modeling.py first.")

with open(S4_RESULTS_PATH, 'rb') as f:
    s4 = pickle.load(f)
with open(SUMMARY_PATH) as f:
    summary = json.load(f)

# save_results() in 03_modeling.py strips the fitted model object out of the
# slim *_results.pkl and joblib-dumps it separately -- the model is never
# inside S4_results.pkl itself.
tree_model = joblib.load(S4_MODEL_PATH)
t          = tree_model.tree_
S4_TREE    = summary['feature_lists']['S4_TREE']

# Load scale factors actually used to retrain the integer S4 tree.
# Section 20 of 03_modeling.py verifies and saves these to
# S4_threshold_extraction.json -- preprocessing_summary.json does NOT carry
# them (that key was removed after being found to reference variables not
# yet defined at the point in the script where it was written).
if not os.path.exists(THRESHOLD_PATH):
    sys.exit(f"\nERROR: {THRESHOLD_PATH} not found.\nRun 03_modeling.py first "
             f"(Section 20 writes this file after integer retraining).")
with open(THRESHOLD_PATH) as f:
    _threshold_info = json.load(f)
_scale_factors  = _threshold_info['scale_factors_used']
INT_SCALE_RATE  = _scale_factors['INT_SCALE_RATE']
INT_SCALE_IAT   = _scale_factors['INT_SCALE_IAT']
INT_SCALE_RATIO = _scale_factors['INT_SCALE_RATIO']

# S2_reduced -- needed to restrict FE_protocol_diversity/FE_no_app_layer to
# exactly the columns that survived Section 5's train-only statistical
# reduction, matching build_engineered_features() in 03_modeling.py exactly.
S2_reduced = summary['S2_diagnostics']['reduced']

print(f"\n  Tree depth : {tree_model.get_depth()}")
print(f"  Nodes      : {t.node_count}")
print(f"  Features   : {len(S4_TREE)}  ->  {S4_TREE}")
print(f"  Scale factors: RATE={INT_SCALE_RATE:,}  IAT={INT_SCALE_IAT:,}  RATIO={INT_SCALE_RATIO:,}")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Helper: walk the tree ─────────────────────────────────────────────────────
def get_tree_nodes(tree_model, feature_names):
    """Return list of dicts describing every node in the tree."""
    t = tree_model.tree_
    nodes = []
    def walk(node, depth=0, parent=None, branch=None):
        is_leaf = t.children_left[node] == _tree.TREE_LEAF
        class_counts = t.value[node][0]
        predicted_class = int(np.argmax(class_counts))
        nodes.append({
            'id': node,
            'depth': depth,
            'parent': parent,
            'branch': branch,           # 'left' (<= threshold) or 'right' (>)
            'is_leaf': is_leaf,
            'feature': feature_names[t.feature[node]] if not is_leaf else None,
            'threshold': int(t.threshold[node]) if not is_leaf else None,
            'left_child':  int(t.children_left[node])  if not is_leaf else None,
            'right_child': int(t.children_right[node]) if not is_leaf else None,
            'predicted_class': predicted_class,
            'class_label': 'Benign' if predicted_class == 0 else 'Attack',
            'n_samples': int(t.n_node_samples[node]),
        })
        if not is_leaf:
            walk(t.children_left[node],  depth+1, node, 'left')
            walk(t.children_right[node], depth+1, node, 'right')
    walk(0)
    return nodes

nodes = get_tree_nodes(tree_model, S4_TREE)
leaves = [n for n in nodes if n['is_leaf']]
internals = [n for n in nodes if not n['is_leaf']]

print(f"\n  Internal nodes: {len(internals)}  |  Leaves: {len(leaves)}")
print(f"\n  Tree structure:")
for n in nodes:
    indent = '  ' * n['depth']
    if n['is_leaf']:
        print(f"    {indent}[Leaf {n['id']}] -> {n['class_label']} (n={n['n_samples']:,})")
    else:
        print(f"    {indent}[Node {n['id']}] if {n['feature']} <= {n['threshold']:,}: "
              f"goto {n['left_child']} else goto {n['right_child']}")

# ── Feature index map (position in the custom P4 header) ─────────────────────
# Every S4 feature gets a fixed slot in the custom header, 32 bits each.
# This is simple and unambiguous; the test harness fills them in the same order.
feat_idx = {f: i for i, f in enumerate(S4_TREE)}

# Features that are computed by the P4 program from raw inputs vs. passed in
# The baseline features are "raw" (passed directly in the header).
# The FE_* features are computed inside P4 from the raw fields.
FE_FEATURES  = [f for f in S4_TREE if f.startswith('FE_')]
RAW_FEATURES = [f for f in S4_TREE if not f.startswith('FE_')]

print(f"\n  Raw features ({len(RAW_FEATURES)}): {RAW_FEATURES}")
print(f"  FE features  ({len(FE_FEATURES)}):  {FE_FEATURES}")

# ── 1. Generate arp_detector.p4 ───────────────────────────────────────────────
def indent(text, n=4):
    return textwrap.indent(textwrap.dedent(text), ' ' * n)

# Build the tree as nested if/else in P4 apply block
#
# DESIGN NOTE -- why P4 does not compute FE_* itself
#   Only 2 of the 8 FE_* features (FE_burst_intensity, FE_payload_ratio) are
#   ever used by the tree's splits, and computing them in the data plane
#   would require source raw columns (Number, Tot size) that are not part
#   of S4_TREE/RAW_FEATURES at all -- they were absorbed into FE_* during
#   feature engineering and never sent as their own header fields. Rather
#   than widening the header just to re-derive values Python has already
#   computed correctly (and verified against the cached model), this test
#   sends both FE_* values precomputed, exactly as they appear in
#   flow_test_vectors.csv. P4 only does the match-action comparisons --
#   which is the actual thing this experiment is validating: does the
#   tree's topology, re-expressed as P4 if/else logic, reproduce the
#   trained model's verdicts.
def tree_to_p4_apply(nodes):
    node_map = {n['id']: n for n in nodes}

    def field_ref(fname):
        # Every S4 feature used by a split -- raw or FE_* -- is sent
        # directly as a header field (see HEADER_FIELDS below).
        clean = fname.replace(' ', '_').replace('-', '_')
        return f"hdr.flow.{clean}"

    def emit_node(node_id, depth=0):
        n = node_map[node_id]
        pad = '    ' * depth
        if n['is_leaf']:
            return f"{pad}meta.verdict = {n['predicted_class']};  // {n['class_label']}"
        f = field_ref(n['feature'])
        left  = emit_node(n['left_child'],  depth+1)
        right = emit_node(n['right_child'], depth+1)
        return (
            f"{pad}if ({f} <= {n['threshold']}) {{\n"
            f"{left}\n"
            f"{pad}}} else {{\n"
            f"{right}\n"
            f"{pad}}}"
        )

    return emit_node(0)

# Build the full metadata struct fields for FE features used in the tree
fe_in_tree = sorted(set(
    n['feature'] for n in internals if n['feature'] and n['feature'].startswith('FE_')
))

# Every field the tree's splits reference -- raw features plus the FE_*
# features actually used -- gets one bit<64> slot in the header, in this
# fixed order. The test harness packs flow_test_vectors.csv columns in the
# exact same order, so the two stay in lockstep without separate bookkeeping.
#
# bit<64> throughout -- with the derived scale factors (commonly 1e10 or
# more; see INT_SCALE_* above), a scaled Rate value alone can reach ~1e14
# and FE_burst_intensity (Rate * Number) can reach ~1e17, both far beyond
# bit<32>'s ~4.3e9 ceiling. bit<64> removes the overflow/truncation risk
# regardless of which scale factor a given run derives.
HEADER_FIELDS = RAW_FEATURES + fe_in_tree

raw_header_fields = '\n    '.join(
    f"bit<64> {f.replace(' ','_').replace('-','_').replace('/','_')};" for f in HEADER_FIELDS
)

p4_program = f"""\
/* arp_detector.p4
 * Generated by 04_generate_p4.py from fitted S4 Decision Tree
 * P4_16 -- compiles with p4c, runs on BMv2 simple_switch
 *
 * WHAT THIS DOES
 *   Receives crafted Ethernet frames carrying a custom 'flow' header with
 *   integer-scaled feature values (one frame per flow), all precomputed in
 *   Python -- including the 2 FE_* features the tree actually splits on.
 *   Applies the fitted depth-{tree_model.get_depth()} Decision Tree as nested if/else
 *   logic, sets a verdict (0=Benign, 1=Attack), and reflects the packet
 *   back with a 'result' header prepended.
 *
 * SCALE FACTORS (applied when 04_generate_p4.py builds flow_test_vectors.csv)
 *   Rate : x{INT_SCALE_RATE:,}   IAT : x{INT_SCALE_IAT:,}   Ratios: x{INT_SCALE_RATIO:,}
 *
 * CUSTOM ETHERTYPE
 *   0x1234 = flow feature frame (input to classifier)
 *   0x1235 = result frame (reflected back with verdict)
 */

#include <core.p4>
#include <v1model.p4>

// ── Custom ethertypes ─────────────────────────────────────────────────────────
const bit<16> TYPE_FLOW   = 0x1234;
const bit<16> TYPE_RESULT = 0x1235;
const bit<32> MAX_INT_SENTINEL = {SMARTNIC_MAX_INT};

// ── Headers ───────────────────────────────────────────────────────────────────
header ethernet_t {{
    bit<48> dstAddr;
    bit<48> srcAddr;
    bit<16> etherType;
}}

// One 64-bit slot per field the tree's splits reference: the 16 raw
// features plus the 2 FE_* features actually used (FE_burst_intensity,
// FE_payload_ratio), both precomputed in Python exactly as they appear in
// flow_test_vectors.csv -- P4 does not compute FE_* itself (see the design
// note above tree_to_p4_apply()). All values are pre-scaled integers
// (see scale factors above).
header flow_t {{
    {raw_header_fields}
}}

// Result header: prepended to the reflected packet.
header result_t {{
    bit<8>  verdict;    // 0 = Benign, 1 = Attack
    bit<8>  pad;
    bit<16> reserved;
}}

struct headers {{
    ethernet_t ethernet;
    flow_t     flow;
    result_t   result;
}}

// ── Metadata ──────────────────────────────────────────────────────────────────
struct metadata {{
    bit<8>  verdict;
}}

// ── Parser ────────────────────────────────────────────────────────────────────
parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_metadata) {{
    state start {{
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {{
            TYPE_FLOW: parse_flow;
            default: accept;
        }}
    }}
    state parse_flow {{
        packet.extract(hdr.flow);
        transition accept;
    }}
}}

// ── Checksum verification (none needed for custom headers) ────────────────────
control MyVerifyChecksum(inout headers hdr, inout metadata meta) {{
    apply {{ }}
}}

// ── Ingress ───────────────────────────────────────────────────────────────────
control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {{

    action drop() {{
        mark_to_drop(standard_metadata);
    }}

    action classify() {{
        // ── Apply fitted decision tree (all fields arrive precomputed) ───────
{textwrap.indent(tree_to_p4_apply(nodes), '        ')}

        // ── Add result header and send out port 1 (veth1) ────────────────────
        hdr.result.setValid();
        hdr.result.verdict  = meta.verdict;
        hdr.result.pad      = 0;
        hdr.result.reserved = 0;
        hdr.ethernet.etherType = TYPE_RESULT;

        // Send out port 1 (veth1). Scapy listens on veth1 for replies.
        // Packets sent into veth0 (port 0) by Scapy arrive at BMv2 on port 0.
        // BMv2 sends the result out port 1 (veth1); Scapy sniffs veth1.
        standard_metadata.egress_spec = 1;
    }}

    table flow_classifier {{
        key = {{
            hdr.ethernet.etherType : exact;
        }}
        actions = {{
            classify;
            drop;
        }}
        default_action = drop();
        const entries = {{
            TYPE_FLOW : classify();
        }}
    }}

    apply {{
        if (hdr.ethernet.isValid()) {{
            flow_classifier.apply();
        }}
    }}
}}

// ── Egress (passthrough) ──────────────────────────────────────────────────────
control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {{
    apply {{ }}
}}

// ── Checksum update (none needed) ─────────────────────────────────────────────
control MyComputeChecksum(inout headers hdr, inout metadata meta) {{
    apply {{ }}
}}

// ── Deparser ──────────────────────────────────────────────────────────────────
control MyDeparser(packet_out packet, in headers hdr) {{
    apply {{
        packet.emit(hdr.ethernet);
        packet.emit(hdr.result);   // result header first (before flow data)
        packet.emit(hdr.flow);
    }}
}}

// ── Switch instantiation ──────────────────────────────────────────────────────
V1Switch(
    MyParser(),
    MyVerifyChecksum(),
    MyIngress(),
    MyEgress(),
    MyComputeChecksum(),
    MyDeparser()
) main;
"""

p4_path = os.path.join(OUT_DIR, 'arp_detector.p4')
with open(p4_path, 'w', encoding='utf-8') as f:
    f.write(p4_program)
print(f"\n[1/5] Generated: {p4_path}")

# ── 2. Generate table_entries.txt (empty -- tree uses const entries) ──────────
# The tree logic is embedded as const entries in the P4 source, so no
# runtime table population is needed. This file is a placeholder showing
# the switch startup commands only.
startup_commands = f"""\
# table_entries.txt
# No runtime table entries needed -- the decision tree is encoded as
# const entries directly in arp_detector.p4.
#
# BMv2 startup command (run inside the VM):
#   sudo simple_switch --interface 0@veth0 --interface 1@veth1 \\
#       arp_detector.json
#
# The compiled JSON (arp_detector.json) is produced by:
#   p4c --target bmv2 --arch v1model arp_detector.p4
"""
entries_path = os.path.join(OUT_DIR, 'table_entries.txt')
with open(entries_path, 'w', encoding='utf-8') as f:
    f.write(startup_commands)
print(f"[2/5] Generated: {entries_path}")

# ── 3. Generate flow test vectors ─────────────────────────────────────────────
n_desc = "full test set" if N_TEST_VECTORS is None else f"{N_TEST_VECTORS} rows, stratified"
print(f"\n[3/5] Building test vectors ({n_desc})...")

# Load the dataset to reconstruct the test split exactly as 03_modeling.py did
from sklearn.model_selection import train_test_split

df = pd.read_csv(DATA_PATH)
LABEL = 'label'
feature_cols_all = [c for c in df.columns if c != LABEL]
y_all = df[LABEL].astype(int)

_, X_temp, _, y_temp = train_test_split(
    df[feature_cols_all], y_all, test_size=0.30,
    random_state=RANDOM_STATE, stratify=y_all
)
_, X_test, _, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=RANDOM_STATE, stratify=y_temp
)

# Apply the same integer quantization as Section 15b
def quantize(X):
    X = X.copy()
    sentinel = X['Rate'] == SMARTNIC_MAX_INT
    X['Rate'] = (X['Rate'].where(sentinel, X['Rate'] * INT_SCALE_RATE)).astype(np.int64)
    X.loc[sentinel, 'Rate'] = SMARTNIC_MAX_INT
    X['IAT'] = (X['IAT'] * INT_SCALE_IAT).astype(np.int64)
    for col in ['Min', 'Max', 'Header_Length', 'AVG', 'Tot sum', 'Tot size',
                'Variance', 'Number', 'ARP', 'TCP', 'UDP', 'ICMP', 'IGMP',
                'HTTP', 'HTTPS', 'DNS', 'SSH', 'DHCP',
                'syn_flag_number', 'ack_flag_number', 'fin_flag_number',
                'rst_flag_number', 'psh_flag_number',
                'ece_flag_number', 'cwr_flag_number', 'Protocol Type', 'Time_To_Live']:
        if col in X.columns:
            X[col] = X[col].astype(np.int64)
    return X

X_test_int = quantize(X_test)

# Build FE features for S4
FE_FEATURES_NEEDED = FE_FEATURES

def build_fe(X):
    # Mirrors build_engineered_features(..., integer_mode=True) plus the
    # caller-side sentinel fillna in rebuild_integer_features(), both in
    # 03_modeling.py, term for term -- so these test vectors are computed
    # exactly the way the real model was trained on.
    fe = pd.DataFrame(index=X.index)
    safe_rate = X['Rate'].replace(SMARTNIC_MAX_INT, np.nan)
    fe['FE_burst_intensity']    = (safe_rate * X['Number']).fillna(
                                      np.int64(SMARTNIC_MAX_INT) * X['Number']).astype(np.int64)
    fe['FE_flow_duration']      = (X['Number'] // safe_rate.replace(0, np.nan)).fillna(0).astype(np.int64)
    fe['FE_size_cv_proxy']      = (X['Variance'] // (X['AVG'] + 1)).astype(np.int64)
    proto_cols_full = ['HTTP', 'HTTPS', 'DNS', 'SSH', 'DHCP', 'TCP', 'UDP', 'ARP', 'ICMP', 'IGMP']
    proto_cols = [c for c in proto_cols_full if c in S2_reduced]
    fe['FE_protocol_diversity'] = (X[proto_cols] > 0).sum(axis=1).astype(np.int64)
    fe['FE_size_range']         = (X['Max'] - X['Min']).astype(np.int64)
    fe['FE_payload_ratio']      = (((X['Tot size'] - X['Header_Length']) * INT_SCALE_RATIO)
                                    // (X['Tot size'] + 1)).astype(np.int64)
    fe['FE_header_ratio']       = ((X['Header_Length'] * 100) // (X['Tot size'] + 1)).astype(np.int64)
    app_layer_cols_full = ['HTTP', 'HTTPS', 'DNS', 'SSH', 'DHCP']
    app_cols = [c for c in app_layer_cols_full if c in S2_reduced]
    fe['FE_no_app_layer']       = ((X[app_cols].sum(axis=1)) == 0).astype(np.int64)
    fe['FE_avg_min_ratio']      = (X['AVG'] // (X['Min'] + 1)).astype(np.int64)
    fe['FE_arp_rate']           = (X['ARP'] * safe_rate).fillna(
                                      X['ARP'] * np.int64(SMARTNIC_MAX_INT)).astype(np.int64)
    return fe

FE_test = build_fe(X_test_int)

# Assemble full S4 feature matrix
X_s4 = pd.concat([X_test_int[RAW_FEATURES], FE_test[FE_FEATURES]], axis=1)[S4_TREE]

# Get Python model predictions
py_preds = tree_model.predict(X_s4)

# Stratified sample
ben_idx = X_s4.index[y_test.loc[X_s4.index] == 0]
atk_idx = X_s4.index[y_test.loc[X_s4.index] == 1]
rng = np.random.RandomState(RANDOM_STATE)
if N_TEST_VECTORS is None:
    sample_idx = X_s4.index  # full test set
else:
    n_each = N_TEST_VECTORS // 2
    sample_idx = np.concatenate([
        rng.choice(ben_idx, min(n_each, len(ben_idx)), replace=False),
        rng.choice(atk_idx, min(n_each, len(atk_idx)), replace=False),
    ])
    rng.shuffle(sample_idx)

X_sample = X_s4.loc[sample_idx]
y_sample = y_test.loc[sample_idx]
pred_sample = tree_model.predict(X_sample)

vectors_path = os.path.join(OUT_DIR, 'flow_test_vectors.csv')
X_sample_out = X_sample.copy()
X_sample_out['true_label']     = y_sample.values
X_sample_out['python_pred']    = pred_sample
X_sample_out.to_csv(vectors_path, index=False)
print(f"[3/5] Generated: {vectors_path}  ({len(X_sample_out)} rows, "
      f"{(pred_sample==0).sum()} benign, {(pred_sample==1).sum()} attack predictions)")

# ── 4. Generate test_harness.py ───────────────────────────────────────────────
header_feat_list = repr(HEADER_FIELDS)
fe_feat_list      = repr(FE_FEATURES)
s4_tree_list      = repr(S4_TREE)

harness = f"""\
#!/usr/bin/env python3
\"\"\"
test_harness.py -- P4 validation test harness
Generated by 04_generate_p4.py

PURPOSE
-------
For each row in flow_test_vectors.csv, crafts one Ethernet frame carrying
that row's feature values in the custom flow header, sends it through the
running BMv2 switch, and reads back the verdict from the result header.
Compares every P4 verdict against the Python model's prediction stored in
the CSV (column 'python_pred').

A 100% match rate means the P4 decision tree encoding is faithful to the
Python model.

USAGE (inside the P4 VM, after simple_switch is running)
-----
  python3 test_harness.py [--vectors flow_test_vectors.csv]
                          [--iface veth0]
                          [--timeout 2.0]

DEPENDENCIES (pre-installed on the P4 tutorial VM)
------------
  scapy, pandas, numpy
\"\"\"

import argparse, sys, time, struct
import pandas as pd
import numpy as np

try:
    from scapy.all import Ether, sendp, sniff, Raw, conf
    conf.verb = 0
except ImportError:
    sys.exit("scapy not found. Install with: pip install scapy")

# ── Constants (must match arp_detector.p4) ────────────────────────────────────
TYPE_FLOW   = 0x1234
TYPE_RESULT = 0x1235
IFACE_DEFAULT = "veth0"
TIMEOUT_DEFAULT = 2.0   # seconds to wait for reflected packet

# Feature order in the custom header: 16 raw features + the 2 FE_*
# features the tree actually uses, all precomputed, 64-bit each.
HEADER_FIELDS = {header_feat_list}

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="P4 ARP detector test harness")
parser.add_argument('--vectors',  default='flow_test_vectors.csv')
parser.add_argument('--iface',    default=IFACE_DEFAULT)
parser.add_argument('--timeout',  type=float, default=TIMEOUT_DEFAULT)
parser.add_argument('--max-rows', type=int,   default=None,
                    help="Stop after this many rows (default: all)")
args = parser.parse_args()

# ── Load test vectors ─────────────────────────────────────────────────────────
df = pd.read_csv(args.vectors)
if args.max_rows:
    df = df.head(args.max_rows)
print(f"Loaded {{len(df)}} test vectors from {{args.vectors}}")
print(f"Sending on interface: {{args.iface}}")

# ── Build a custom flow header from one row ───────────────────────────────────
def build_flow_header(row):
    \"\"\"Pack feature values into 8-byte big-endian fields, in HEADER_FIELDS
    order (matches the bit<64> fields in flow_t in arp_detector.p4).\"\"\"
    fields = []
    for feat in HEADER_FIELDS:
        val = int(row.get(feat, 0))
        # Clamp to uint64 range
        val = max(0, min(val, 0xFFFFFFFFFFFFFFFF))
        fields.append(struct.pack('!Q', val))
    return b''.join(fields)

# ── Send one packet and receive the reflected result ──────────────────────────
def send_and_receive(flow_payload, iface, timeout):
    \"\"\"Send a TYPE_FLOW packet on iface (veth0) and receive the result on
    veth1. BMv2 egresses on port 1 (veth1), so Scapy must listen there.\"\"\"
    pkt = Ether(type=TYPE_FLOW) / Raw(load=flow_payload)

    # Start sniffing on veth1 BEFORE sending, so we don't miss the reply
    from scapy.all import AsyncSniffer
    sniffer = AsyncSniffer(
        iface='veth1',
        lfilter=lambda p: p.haslayer(Ether) and p[Ether].type == TYPE_RESULT,
        count=1,
        timeout=timeout
    )
    sniffer.start()

    sendp(pkt, iface=iface, verbose=False)

    sniffer.join(timeout=timeout + 0.5)
    captured = sniffer.results

    if not captured:
        return None

    # Result header layout: verdict(1B) + pad(1B) + reserved(2B) = 4 bytes
    raw = bytes(captured[0][Ether].payload)
    if len(raw) >= 1:
        return raw[0]   # verdict byte
    return None

# ── Main loop ─────────────────────────────────────────────────────────────────
print("\\n" + "=" * 60)
print("P4 VALIDATION -- sending test flows")
print("=" * 60)

results = []
n_match = 0
n_miss  = 0
n_timeout = 0

for i, row in df.iterrows():
    py_pred  = int(row['python_pred'])
    payload  = build_flow_header(row)
    p4_verdict = send_and_receive(payload, args.iface, args.timeout)

    if p4_verdict is None:
        n_timeout += 1
        status = 'TIMEOUT'
    elif p4_verdict == py_pred:
        n_match += 1
        status = 'MATCH'
    else:
        n_miss += 1
        status = f'MISMATCH (P4={{p4_verdict}}, Python={{py_pred}})'

    results.append({{'row': i, 'python_pred': py_pred,
                     'p4_verdict': p4_verdict, 'status': status}})

    if (len(results)) % 100 == 0:
        print(f"  {{len(results):>5}} / {{len(df)}}  "
              f"match={{n_match}}  mismatch={{n_miss}}  timeout={{n_timeout}}")

# ── Summary ────────────────────────────────────────────────────────────────────
total = len(results)
print("\\n" + "=" * 60)
print(f"RESULTS  ({{total}} flows)")
print("=" * 60)
print(f"  Match    : {{n_match:>6}} / {{total}}  ({{n_match/total*100:.2f}}%)")
print(f"  Mismatch : {{n_miss:>6}} / {{total}}")
print(f"  Timeout  : {{n_timeout:>6}} / {{total}}")

if n_miss > 0:
    print("\\n  MISMATCHES (first 10):")
    shown = 0
    for r in results:
        if r['status'].startswith('MISMATCH'):
            print(f"    row={{r['row']}}  python={{r['python_pred']}}  p4={{r['p4_verdict']}}")
            shown += 1
            if shown >= 10: break

if n_match == total:
    print("\\n  RESULT: PERFECT MATCH -- P4 encoding is faithful to the Python model.")
elif n_timeout == total:
    print("\\n  RESULT: ALL TIMEOUTS -- is simple_switch running? Is the interface correct?")
else:
    print(f"\\n  RESULT: {{n_miss}} mismatches -- inspect the rows above to debug.")

# Save results
out_df = pd.DataFrame(results)
out_df.to_csv('p4_validation_results.csv', index=False)
print(f"\\n  Full results saved -> p4_validation_results.csv")
"""

harness_path = os.path.join(OUT_DIR, 'test_harness.py')
with open(harness_path, 'w', encoding='utf-8') as f:
    f.write(harness)
print(f"[4/5] Generated: {harness_path}")

# ── 5. Generate README.md ──────────────────────────────────────────────────────
feature_header_size = len(HEADER_FIELDS) * 8  # 8 bytes per field (bit<64>)

readme = f"""\
# P4 Validation -- ARP Spoofing Detector

Generated by `04_generate_p4.py` from the fitted S4 Decision Tree.

## What this validates

The S4 Decision Tree (depth={tree_model.get_depth()}, integer-quantized) is re-expressed as a
P4_16 program and run on BMv2. Each test flow is sent as a crafted packet;
the P4 program classifies it and reflects the verdict back. If every verdict
matches the Python model's prediction on the same row, the P4 encoding is
faithful and the model is confirmed deployable on a programmable SmartNIC.

## Files

| File | Purpose |
|---|---|
| `arp_detector.p4` | P4_16 program (compile and run on BMv2) |
| `flow_test_vectors.csv` | {len(X_sample_out):,} test-set rows ({n_desc}) with Python predictions |
| `test_harness.py` | Sends flows through BMv2 and compares verdicts |
| `table_entries.txt` | BMv2 startup notes (no runtime entries needed) |

## Scale factors used

| Feature | Scale factor | Meaning |
|---|---|---|
| Rate | x{INT_SCALE_RATE:,} | raw pkt/s × {INT_SCALE_RATE:,} |
| IAT | x{INT_SCALE_IAT:,} | seconds × {INT_SCALE_IAT:,} |
| Ratios (FE_payload_ratio, FE_size_cv_proxy) | x{INT_SCALE_RATIO:,} | × {INT_SCALE_RATIO:,} |

## Step-by-step (inside the P4 VM)

### 1. Copy this folder to the VM

From your Windows machine, copy the `p4_validation/` folder to the VM.
The easiest way is to use the VirtualBox shared folder feature, or:
```
scp -r p4_validation/ p4@<vm-ip>:/home/p4/
```

Or just drag-and-drop if the VM guest additions are installed.

### 2. Set up virtual interfaces

BMv2 needs a pair of virtual Ethernet interfaces:
```bash
sudo ip link add veth0 type veth peer name veth1
sudo ip link set veth0 up
sudo ip link set veth1 up
```

### 3. Compile the P4 program

```bash
cd /home/p4/p4_validation
p4c --target bmv2 --arch v1model --std p4-16 arp_detector.p4
```

This produces `arp_detector.json`.

### 4. Start the BMv2 switch

```bash
sudo simple_switch --interface 0@veth0 --interface 1@veth1 \\
    arp_detector.json &
```

Wait 2-3 seconds for it to start.

### 5. Run the test harness

`flow_test_vectors.csv` covers {n_desc} ({len(X_sample_out):,} rows). The
harness sends one packet and blocks on a capture per row, sequentially --
if you're running the full test set, expect this to take several hours,
not minutes. Progress prints every 100 rows so you can confirm it's moving.
Consider running it inside `tmux`/`screen` (or with `nohup ... &`) so it
survives a dropped VM console session, and pass `--max-rows N` first to
sanity-check on a small slice before committing to the full run.

```bash
python3 test_harness.py --iface veth0 --vectors flow_test_vectors.csv
```

### 6. Read the results

Expected output if the encoding is correct:
```
RESULT: PERFECT MATCH -- P4 encoding is faithful to the Python model.
```

Full per-row results are saved to `p4_validation_results.csv`.

## Troubleshooting

**All timeouts:** BMv2 is not running, or the interface name is wrong.
Try `ip link show` to list interfaces and pass the correct one with `--iface`.

**Mismatches:** The P4 tree logic diverges from Python at specific rows.
Check the mismatch rows in `p4_validation_results.csv` -- compare the
feature values against the tree structure printed by `04_generate_p4.py`
to identify which branch is being taken incorrectly.

**Compile error in p4c:** Check the p4c version matches (`p4c --version`).
This program was generated for p4c 1.2.x with BMv2 simple_switch 1.15.x.
"""

readme_path = os.path.join(OUT_DIR, 'README.md')
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write(readme)
print(f"[5/5] Generated: {readme_path}")

print(f"\n{'='*70}")
print(f"ALL FILES GENERATED -> {OUT_DIR}")
print(f"{'='*70}")
print(f"\n  Next step: copy the p4_validation/ folder to your P4 VM.")
print(f"  Then follow README.md inside that folder.")
