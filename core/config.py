"""
Central configuration: LLM model name, all output Path constants,
and ensure_dirs() to create them at startup.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
 
load_dotenv()
 
BASE_DIR = Path(__file__).parent.parent
 
PROFILES_DIR = BASE_DIR / "data" / "profiles"
STTM_DIR = BASE_DIR / "data" / "sttm"
BRONZE_DIR = BASE_DIR / "data" / "bronze_layer"
SILVER_DIR = BASE_DIR / "data" / "silver_layer"
GOLD_DIR = BASE_DIR / "data" / "gold_layer"
REPORTS_DIR = BASE_DIR / "data" / "reports"
AUDIT_DIR = BASE_DIR / "data" / "audit_logs"
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
 
# --- LLM provider config ---
# Groq exposes an OpenAI-compatible endpoint, so the standard OpenAI()
# client class is reused with a different base_url + key.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
 
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
 
 
def ensure_dirs() -> None:
    """Create all data subdirectories with exist_ok=True."""
    for d in (
        PROFILES_DIR,
        STTM_DIR,
        BRONZE_DIR,
        SILVER_DIR,
        GOLD_DIR,
        REPORTS_DIR,
        AUDIT_DIR,
        UPLOADS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
