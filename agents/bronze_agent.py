"""
Bronze Agent.
 
Bronze is the fidelity layer: no business logic, no joins. Only type
casting, column renaming to snake_case, and two metadata columns.
Row count in = row count out. The LLM (Groq) generates the DuckDB SQL;
this agent executes it deterministically.
"""
import re
from datetime import datetime
from pathlib import Path
 
import duckdb
import pandas as pd
 
from core.config import BRONZE_DIR
from core.audit import AuditLogger
from core.llm import call_llm
from core import parquet_safety
 
 
def _extract_sql(text: str) -> str:
    match = re.search(r"```sql\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"(SELECT\s+.+?;)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""
 
 
def run(input_files: list[str], sttm_path: str, run_id: str) -> list[str]:
    audit = AuditLogger(run_id)
    sttm_df = pd.read_csv(sttm_path)
    written_paths: list[str] = []
 
    for fp in input_files:
        table_name = Path(fp).stem
        table_rules = sttm_df[sttm_df["source_table"] == table_name]
        sample_df = pd.read_csv(fp, nrows=3)
 
        prompt = (
            f"CSV file path: {fp}\n"
            f"Table: {table_name}\n\n"
            f"STTM rules for this table:\n{table_rules.to_string(index=False)}\n\n"
            f"Column type sample (first 3 rows):\n{sample_df.to_string(index=False)}\n\n"
            "Generate ONE DuckDB SELECT SQL using read_csv_auto() with "
            "all_varchar=true. Use TRY_CAST() for type_cast rules. Map: "
            "datetime->TIMESTAMP, float->DOUBLE, int->INTEGER, str->VARCHAR. "
            "Do NOT add metadata columns in SQL (added afterward in Python).\n"
            "Respond with a ```sql fenced code block containing ONLY the SQL."
        )
 
        df = None
        sql_used = None
        try:
            llm_text = call_llm(prompt, max_tokens=1500)
            sql = _extract_sql(llm_text)
            if sql:
                conn = duckdb.connect()
                df = conn.execute(sql).df()
                sql_used = sql
        except Exception:
            df = None
 
        if df is None:
            # Fallback: passthrough
            df = pd.read_csv(fp)
            sql_used = "FALLBACK: pd.read_csv passthrough"
 
        df["_load_timestamp"] = datetime.utcnow().isoformat() + "Z"
        df["_source_file"] = str(fp)
 
        out_path = BRONZE_DIR / f"{table_name}_bronze_{run_id[:8]}.parquet"
        try:
            df = parquet_safety.sanitize_for_parquet(df)
        except Exception:
            # fallback sanitizer: coerce non-serializable object values to str
            for col in df.columns:
                if df[col].dtype == object:
                    sample = df[col].dropna()
                    if sample.empty:
                        continue
                    first_val = sample.iloc[0]
                    if not isinstance(first_val, (str, int, float, bool)):
                        df[col] = df[col].astype(str)
        df.to_parquet(out_path, index=False)
        written_paths.append(str(out_path))
 
        audit.log(
            "bronze_agent",
            "table_processed",
            table=table_name,
            output_path=str(out_path),
            row_count=len(df),
            sql=sql_used,
        )
 
    return written_paths