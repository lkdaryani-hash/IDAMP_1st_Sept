"""
Silver Agent.
 
Silver is the quality layer: reads Bronze Parquet, applies cleansing
rules (STTM-driven), row count may decrease via dedup, and adds a
surrogate integer key. LLM (Groq) generates the DuckDB SQL.
"""
import re
from pathlib import Path
 
import duckdb
import pandas as pd
 
from core.config import SILVER_DIR
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
 
 
def _base_name(bronze_table_name: str, run_id: str) -> str:
    return bronze_table_name.replace(f"_bronze_{run_id[:8]}", "")
 
 
def run(bronze_paths: list[str], sttm_path: str, run_id: str) -> list[str]:
    audit = AuditLogger(run_id)
    sttm_df = pd.read_csv(sttm_path)
    written_paths: list[str] = []
    conn = duckdb.connect()
 
    for fp in bronze_paths:
        raw_stem = Path(fp).stem  # e.g. 'sales data_bronze_a1b2c3d4' (may contain spaces)
        # sanitize a safe DuckDB table alias (no spaces or special chars)
        table_alias = re.sub(r"\W+", "_", raw_stem)
        base_name = _base_name(raw_stem, run_id)
 
        conn.execute(
            f"CREATE OR REPLACE TABLE {table_alias} AS "
            f"SELECT * FROM read_parquet('{fp}')"
        )
 
        df_preview = conn.execute(f"SELECT * FROM {table_alias} LIMIT 200").df()
        categorical_distincts = {}
        for col in df_preview.select_dtypes(include="object").columns:
            categorical_distincts[col] = df_preview[col].dropna().unique()[:10].tolist()
 
        table_rules = sttm_df[sttm_df["source_table"] == base_name]
 
        prompt = (
            f"Table (DuckDB alias): {table_alias}\n"
            f"Schema: {list(df_preview.columns)}\n"
            f"Categorical distinct value samples: {categorical_distincts}\n\n"
            f"STTM rules for this table:\n{table_rules.to_string(index=False)}\n\n"
            "Generate ONE DuckDB SELECT SQL against this table implementing:\n"
            "- standardize_date: COALESCE(TRY_STRPTIME(col,'%Y-%m-%d'), "
            "TRY_STRPTIME(col,'%m/%d/%Y'))::VARCHAR\n"
            "- normalize_category: CASE WHEN with patterns "
            "('elec%'->'Electronics', 'apparel%'/'cloth%'->'Apparel', "
            "'beauty%'/'cosmetic%'->'Beauty', 'sport%'->'Sports', "
            "'home%'/'kitchen%'->'Home & Kitchen')\n"
            "- normalize_segment: ('prem%'->'Premium', 'bgt%'/'budg%'->'Budget', "
            "'stand%'/'std%'->'Standard', 'loyal%'->'Loyal')\n"
            "- null_handler drop_nulls: WHERE col IS NOT NULL\n"
            "- null_handler fill_median: COALESCE(col, median subquery)\n"
            "- dedup: SELECT DISTINCT\n"
            "- surrogate_key: IGNORE this rule entirely -- do NOT generate a "
            "UUID, hash, or any additional key column for it. A numeric "
            "surrogate key is added automatically in Python after your SQL "
            "runs, so any column with this rule should just be omitted.\n"
            "- passthrough: SELECT col AS col unchanged\n"
            f"Select FROM {table_alias}.\n"
            "Respond with a ```sql fenced code block containing ONLY the SQL."
        )
 
        df = None
        sql_used = None
        try:
            llm_text = call_llm(prompt, max_tokens=2500)
            sql = _extract_sql(llm_text)
            if sql:
                df = conn.execute(sql).df()
                sql_used = sql
        except Exception:
            df = None
 
        if df is None:
            df = pd.read_parquet(fp).drop_duplicates()
            sql_used = "FALLBACK: pd.read_parquet + drop_duplicates"
 
        df.insert(0, f"pk_{base_name}_silver_id", range(1, len(df) + 1))
        try:
            df = parquet_safety.sanitize_for_parquet(df)
        except Exception:
            for col in df.columns:
                if df[col].dtype == object:
                    sample = df[col].dropna()
                    if sample.empty:
                        continue
                    first_val = sample.iloc[0]
                    if not isinstance(first_val, (str, int, float, bool)):
                        df[col] = df[col].astype(str)
 
        out_path = SILVER_DIR / f"{base_name}_silver_{run_id[:8]}.parquet"
        df.to_parquet(out_path, index=False)
        written_paths.append(str(out_path))
 
        audit.log(
            "silver_agent",
            "table_processed",
            table=base_name,
            output_path=str(out_path),
            row_count=len(df),
            sql=sql_used,
        )
 
    return written_paths