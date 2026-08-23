import sys
from pathlib import Path

# Make the `camera` package importable regardless of pytest invocation dir.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
