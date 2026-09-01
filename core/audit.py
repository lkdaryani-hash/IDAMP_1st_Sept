"""
Append-only JSONL logger. One file per run. Every agent logs its
key actions here for traceability.
"""
import json
from datetime import datetime
 
from core.config import AUDIT_DIR
 
 
class AuditLogger:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.log_path = AUDIT_DIR / f"{run_id}.jsonl"
 
    def log(self, agent: str, action: str, **kwargs) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "run_id": self.run_id,
            "agent": agent,
            "action": action,
            **kwargs,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
 
    def get_logs(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        with open(self.log_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


