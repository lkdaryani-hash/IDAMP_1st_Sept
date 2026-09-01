"""
PipelineState dataclass -- the pipeline's shared memory baton.
Each phase reads from it and writes its outputs back.
"""
from dataclasses import dataclass, field
from typing import Optional
 
 
@dataclass
class PipelineState:
    run_id: str  # UUID for this pipeline run
    input_files: list[str] = field(default_factory=list)  # Paths to original CSV files
    business_intent: str = ""  # The user's business question
 
    profile_path: Optional[str] = None  # Phase 1: profiler output JSON
    bronze_sttm_path: Optional[str] = None
    bronze_paths: list[str] = field(default_factory=list)  # Phase 2: Bronze parquet files
 
    silver_sttm_path: Optional[str] = None
    silver_paths: list[str] = field(default_factory=list)
 
    gold_sttm_path: Optional[str] = None
    gold_paths: list[str] = field(default_factory=list)
 
    report_path: Optional[str] = None  # Final HTML report
