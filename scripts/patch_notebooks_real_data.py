"""One-off: repoint notebooks 01/02 from synthetic dummy data to the real ASVspoof 2019 LA dataset.
Also clears stale outputs (they were produced from synthetic audio) and repairs invalid JSON in notebook 01."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(p):
    return json.loads(p.read_text(), strict=False)  # strict=False tolerates the raw newline bug in nb01

def save(p, nb):
    for c in nb["cells"]:
        src = "".join(c["source"])
        c["source"] = src.splitlines(keepends=True)
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    p.write_text(json.dumps(nb, indent=1))

def replace(nb, old, new, count_expected=1):
    hits = 0
    for c in nb["cells"]:
        src = "".join(c["source"])
        if old in src:
            c["source"] = src.replace(old, new)
            hits += 1
    assert hits == count_expected, f"expected {count_expected} hit(s) for {old[:50]!r}, got {hits}"

# ---- Notebook 01 ----
p1 = ROOT / "notebooks" / "01_data_exploration.ipynb"
nb1 = load(p1)
replace(nb1, "from src.data.dataset import ASVspoofDataset, generate_dummy_dataset",
             "from src.data.dataset import ASVspoofDataset")
replace(nb1,
"""# Generate synthetic dummy dataset if real dataset is not on disk
dummy_paths = generate_dummy_dataset(num_train=120, num_dev=30, num_eval=30)

train_ds = ASVspoofDataset(
    protocol_path=config.paths.dummy_data_dir / "LA" / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trl.txt",
    audio_dir=config.paths.dummy_data_dir / "LA" / "ASVspoof2019_LA_train" / "flac",
    is_train=True
)""",
"""# Real ASVspoof 2019 LA only. Raises FileNotFoundError if data/LA/ is missing (no synthetic fallback).
train_ds = ASVspoofDataset(
    protocol_path=config.paths.train_protocol,
    audio_dir=config.paths.train_audio_dir,
    is_train=True,
    auto_generate_dummy=False,
)""")
save(p1, nb1)

# ---- Notebook 02 ----
p2 = ROOT / "notebooks" / "02_feature_visualization.ipynb"
nb2 = load(p2)
replace(nb2, "from src.data.dataset import generate_dummy_dataset, ASVspoofDataset",
             "from src.data.dataset import ASVspoofDataset")
replace(nb2,
"""# ── Load dummy samples ─────────────────────────────────────────────────────
generate_dummy_dataset(num_train=60, num_dev=20, num_eval=20)

ds = ASVspoofDataset(
    protocol_path=config.paths.dummy_data_dir / 'LA' / 'ASVspoof2019_LA_cm_protocols' / 'ASVspoof2019.LA.cm.train.trl.txt',
    audio_dir=config.paths.dummy_data_dir / 'LA' / 'ASVspoof2019_LA_train' / 'flac',
    is_train=False  # deterministic crop
)""",
"""# ── Load real ASVspoof 2019 LA samples (no synthetic fallback) ─────────────
ds = ASVspoofDataset(
    protocol_path=config.paths.train_protocol,
    audio_dir=config.paths.train_audio_dir,
    is_train=False,  # deterministic crop
    auto_generate_dummy=False,
)""")
save(p2, nb2)
print("patched notebooks 01 and 02")
