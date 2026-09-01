"""
Profiler Agent.
 
Scans input CSVs, computes per-column quality stats deterministically,
then asks the LLM (Groq) to interpret those stats into actionable
transformation recommendations. Output is a single JSON profile file.
"""
import json
import re
from pathlib import Path
 
import pandas as pd
 
from core.config import PROFILES_DIR
from core.audit import AuditLogger
from core.llm import call_llm
 
DATE_MDY_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
DATE_YMD_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
 
 
def _profile_column(series: pd.Series) -> dict:
    non_null = series.dropna()
    col_info = {
        "dtype": str(series.dtype),
        "null_count": int(series.isna().sum()),
        "null_pct": round(float(series.isna().mean() * 100), 2),
        "unique_count": int(series.nunique(dropna=True)),
        "sample_values": [str(v) for v in non_null.unique()[:5].tolist()],
    }
 
    if pd.api.types.is_numeric_dtype(series) and len(non_null) > 0:
        col_info["min"] = float(non_null.min())
        col_info["max"] = float(non_null.max())
        col_info["mean"] = round(float(non_null.mean()), 4)
 
    quality_flags = []
    if series.dtype == object and len(non_null) > 0:
        str_vals = non_null.astype(str)
        mdy_pct = str_vals.str.match(DATE_MDY_RE).mean()
        ymd_pct = str_vals.str.match(DATE_YMD_RE).mean()
        if mdy_pct > 0.10 and ymd_pct > 0.10:
            quality_flags.append("mixed_date_formats")
 
        avg_len = str_vals.str.len().mean()
        if avg_len < 6:
            quality_flags.append("possible_abbreviations")
 
    col_info["quality_flags"] = quality_flags
    return col_info
 
 
def _find_candidate_join_keys(tables: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    id_cols: dict[str, list[str]] = {}
    for table_name, df in tables.items():
        for col in df.columns:
            if col.endswith("_id"):
                id_cols.setdefault(col, []).append(table_name)
    return {col: tbls for col, tbls in id_cols.items() if len(tbls) >= 2}
 
 
def _build_llm_summary(tables_profile: dict, join_keys: dict) -> str:
    lines = ["Data quality profile summary:\n"]
    for table_name, info in tables_profile.items():
        lines.append(f"Table: {table_name} ({info['row_count']} rows)")
        for col, col_info in info["columns"].items():
            flags = ", ".join(col_info["quality_flags"]) or "none"
            lines.append(
                f"  - {col}: dtype={col_info['dtype']}, "
                f"null_pct={col_info['null_pct']}%, flags=[{flags}], "
                f"samples={col_info['sample_values']}"
            )
    lines.append(f"\nCandidate join keys: {join_keys}")
    lines.append(
        "\nRespond ONLY with JSON containing these keys: "
        "quality_summary (str), critical_issues (list of str), "
        "transformation_recommendations (dict keyed by 'table.column'), "
        "join_key_reasoning (str)."
    )
    return "\n".join(lines)
 
 
def _parse_llm_json(text: str) -> dict:
    match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    payload = match.group(1) if match else text
    return json.loads(payload)
 
 
def profile(file_paths: list[str], run_id: str) -> str:
    audit = AuditLogger(run_id)
    tables: dict[str, pd.DataFrame] = {}
    tables_profile: dict = {}
 
    for fp in file_paths:
        table_name = Path(fp).stem
        df = pd.read_csv(fp, low_memory=False)
        tables[table_name] = df
 
        columns_profile = {col: _profile_column(df[col]) for col in df.columns}
        tables_profile[table_name] = {
            "row_count": len(df),
            "columns": columns_profile,
        }
 
    join_keys = _find_candidate_join_keys(tables)
 
    try:
        summary_prompt = _build_llm_summary(tables_profile, join_keys)
        llm_text = call_llm(summary_prompt, max_tokens=1500)
        llm_analysis = _parse_llm_json(llm_text)
    except Exception as e:
        llm_analysis = {
            "quality_summary": "Rule-based only (LLM unavailable)",
            "critical_issues": [],
            "transformation_recommendations": {},
            "join_key_reasoning": f"LLM call failed: {e}",
        }
 
    result = {
        "run_id": run_id,
        "tables": tables_profile,
        "candidate_join_keys": join_keys,
        "llm_analysis": llm_analysis,
    }
 
    out_path = PROFILES_DIR / f"profile_combined_{run_id[:8]}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
 
    audit.log("profiler", "profile_generated", output_path=str(out_path))
    return str(out_path)
