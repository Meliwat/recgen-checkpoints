from pathlib import Path

MODEL = "EleutherAI/pythia-410m"
LAYER = 12  # middle of Pythia-410m's 24 layers
CONTEXT_LEN = 512
N_WINDOWS = 4096  # 4096 windows x 512 tokens = ~2.1M tokens
SEED = 0

# Log-spaced so early training, where neurons first take shape, is sampled densely.
CHECKPOINT_STEPS = [0, 128, 256, 512, 1000, 2000, 3000, 5000, 8000, 13000, 21000, 34000, 55000, 89000, 143000]
FINAL_STEP = CHECKPOINT_STEPS[-1]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
WEIGHTS_DIR = ROOT / ".weights"  # one checkpoint at a time, deleted after use
