"""Task-specific paths for ReviewPulse v3 ABSA.

Importing this module must remain side-effect free: it defines paths but does
not create directories, load legacy artifacts, or import model code.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ABSA_DATA_DIR = PROJECT_ROOT / "data" / "semeval2014"
ABSA_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "absa"
ABSA_METRICS_DIR = ABSA_OUTPUTS_DIR / "metrics"
ABSA_PLOTS_DIR = ABSA_OUTPUTS_DIR / "plots"
