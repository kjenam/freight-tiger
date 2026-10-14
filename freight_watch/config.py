from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

SHIPMENTS_CSV = DATA_DIR / "shipment_records.csv"
NOTES_CSV = DATA_DIR / "context_notes.csv"
NOTES_STRUCTURED_JSON = DATA_DIR / "notes_structured.json"
SAMPLE_CSV = DATA_DIR / "sample_output_format_v2.csv"

HISTORY_WEEKS = 8
OWN_RISING_MIN_PCT = 0.0
OWN_ABNORMAL_PCT = 10.0
PEER_ABNORMAL_PCT = 20.0

DEFAULT_NOTE_WINDOW_WEEKS = 4
RETRIEVAL_TOP_K = 3

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"
LLM_TEMPERATURE = 0
