"""
STTM Generator.
 
Reads profile/schema context and asks the LLM (Groq) to produce
Source-to-Target Mapping CSVs for Bronze, Silver, and Gold layers.
Agents downstream read these STTMs and execute them deterministically
with DuckDB SQL.
"""
import csv
import io
import json
import re
from pathlib import Path
 
import pandas as pd
 
from core.config import STTM_DIR
from core.audit import AuditLogger
from core.llm import call_llm
 
REQUIRED_COLUMNS = [
    "source_schema",
    "source_table",
    "source_column",
    "target_schema",
    "target_table",
    "target_column",
    "transformation_type",
    "transformation_logic",
]
 
 
def _extract_csv(text: str) -> str:
    # 1. ```csv fenced block (preferred format we asked for)
    match = re.search(r"```csv\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
 
    # 2. any generic fenced block (model sometimes drops the "csv" tag)
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
 
    # 3. no fence at all -- find the header line by looking for the first
    #    required column name and slice from there to the end of the text
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if REQUIRED_COLUMNS[0] in line and REQUIRED_COLUMNS[-1].split("_")[0] in line:
            return "\n".join(lines[i:]).strip()
 
    return text.strip()
 
 
def _validate_and_save(csv_text: str, out_path: Path, raw_llm_text: str = "") -> None:
    reader = csv.DictReader(io.StringIO(csv_text))
    fieldnames = [f.strip() for f in (reader.fieldnames or [])]
    missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    if missing:
        debug_path = out_path.with_suffix(".raw_llm_response.txt")
        debug_path.write_text(raw_llm_text or csv_text, encoding="utf-8")
        raise ValueError(
            f"STTM missing required columns: {missing}. "
            f"Raw LLM response saved to {debug_path} for inspection."
        )
 
    # Re-parse as raw rows (not dict) so we can catch and repair rows where
    # a free-text field -- almost always transformation_logic, the last
    # column -- contains an unquoted comma. The LLM doesn't always quote
    # values like "SUM(price, quantity)", which makes that row parse with
    # more fields than the header. Since transformation_logic is always
    # the last column, any extra fields on a row are merged back into it
    # rather than treated as a hard failure.
    num_cols = len(fieldnames)
    raw_rows = list(csv.reader(io.StringIO(csv_text)))
    header, data_rows = raw_rows[0], raw_rows[1:]
 
    repaired_rows = []
    unrepairable = []
    for i, row in enumerate(data_rows, start=2):  # start=2: line 1 is header
        if len(row) == num_cols:
            repaired_rows.append(row)
        elif len(row) > num_cols:
            fixed = row[: num_cols - 1] + [", ".join(row[num_cols - 1:])]
            repaired_rows.append(fixed)
        elif len(row) > 0:
            # Too few fields -- pad rather than drop, and flag if severely short
            repaired_rows.append(row + [""] * (num_cols - len(row)))
 
    # Re-write with proper quoting so any comma-containing field is safely
    # quoted -- this guarantees pandas (or anything else) can read it back
    # correctly regardless of how the LLM originally formatted it.
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        writer.writerows(repaired_rows)
 
 
def generate_bronze_sttm(profile_path: str, business_intent: str, run_id: str) -> str:
    audit = AuditLogger(run_id)
    with open(profile_path, "r", encoding="utf-8") as f:
        profile_data = json.load(f)
 
    lines = [f"Business intent: {business_intent}\n", "Table profiles:"]
    for table_name, info in profile_data["tables"].items():
        lines.append(f"\nTable: {table_name} ({info['row_count']} rows)")
        for col, col_info in info["columns"].items():
            flags = ", ".join(col_info["quality_flags"]) or "none"
            lines.append(
                f"  - {col}: dtype={col_info['dtype']}, flags=[{flags}], "
                f"samples={col_info['sample_values']}"
            )
 
    lines.append(
        "\nGenerate a CSV with columns: source_schema, source_table, source_column, "
        "target_schema, target_table, target_column, transformation_type, "
        "transformation_logic.\n"
        "Rules to generate: type_cast for every column (infer type from dtype/flags), "
        "plus metadata_inject rows for _load_timestamp (utc_now) and _source_file "
        "(source_path) per table.\n"
        "Respond with a ```csv fenced code block containing ONLY the CSV."
    )
    prompt = "\n".join(lines)
 
    llm_text = call_llm(prompt, max_tokens=1500)
    if not llm_text:
        raise RuntimeError("LLM unavailable -- cannot generate Bronze STTM.")
 
    csv_text = _extract_csv(llm_text)
    out_path = STTM_DIR / f"sttm_bronze_{run_id[:8]}.csv"
    _validate_and_save(csv_text, out_path, raw_llm_text=llm_text)
    audit.log("sttm_generator", "bronze_sttm_generated", output_path=str(out_path))
    return str(out_path)
 
 
def _schema_summary(parquet_paths: list[str]) -> str:
    blocks = []
    for p in parquet_paths:
        df = pd.read_parquet(p)
        blocks.append(
            f"Table (from {Path(p).name}): columns={list(df.columns)}, "
            f"sample_row={df.head(1).to_dict(orient='records')}"
        )
    return "\n".join(blocks)
 
 
def generate_silver_sttm(
    bronze_paths: list[str],
    bronze_sttm_path: str,
    business_intent: str,
    run_id: str,
) -> str:
    audit = AuditLogger(run_id)
    schema_block = _schema_summary(bronze_paths)
    prior_sttm = Path(bronze_sttm_path).read_text(encoding="utf-8")
 
    prompt = (
        f"Business intent: {business_intent}\n\n"
        f"Bronze table schemas:\n{schema_block}\n\n"
        f"Prior Bronze STTM (for context):\n{prior_sttm}\n\n"
        "Generate a Silver STTM CSV with the same 8 required columns "
        "(source_schema, source_table, source_column, target_schema, target_table, "
        "target_column, transformation_type, transformation_logic).\n"
        "Use transformation types: null_handler (drop_nulls / fill_median / "
        "fill_unknown), standardize_date, normalize_category, normalize_segment, "
        "dedup, surrogate_key.\n"
        "Respond with a ```csv fenced code block containing ONLY the CSV."
    )
 
    llm_text = call_llm(prompt, max_tokens=3000)
    if not llm_text:
        raise RuntimeError("LLM unavailable -- cannot generate Silver STTM.")
 
    csv_text = _extract_csv(llm_text)
    out_path = STTM_DIR / f"sttm_silver_{run_id[:8]}.csv"
    _validate_and_save(csv_text, out_path, raw_llm_text=llm_text)
    audit.log("sttm_generator", "silver_sttm_generated", output_path=str(out_path))
    return str(out_path)
 
 
def generate_gold_sttm(
    silver_paths: list[str],
    silver_sttm_path: str,
    business_intent: str,
    run_id: str,
) -> str:
    audit = AuditLogger(run_id)
    schema_block = _schema_summary(silver_paths)
    prior_sttm = Path(silver_sttm_path).read_text(encoding="utf-8")
 
    prompt = (
        f"Business intent: {business_intent}\n\n"
        f"Silver table schemas:\n{schema_block}\n\n"
        f"Prior Silver STTM (for context):\n{prior_sttm}\n\n"
        "Before writing any rows, work out these four things explicitly, using "
        "the ACTUAL table and column names shown in the schemas above (never "
        "placeholders like 'table_a'):\n"
        "1. METRIC: what numeric value answers the business intent, and exactly "
        "which column(s) it comes from (e.g. revenue = SUM(quantity * unit_price), "
        "naming the real quantity and price columns from the schema).\n"
        "2. JOIN: which fact table (the one with transaction-level rows) must be "
        "joined to which dimension table(s), and on which real shared key column.\n"
        "3. TIME WINDOW: if the intent mentions a time period (e.g. 'last N "
        "quarters/months/years'), which real date column to derive the period "
        "from, and the exact filter condition -- do not skip this even if it's "
        "the hardest part.\n"
        "4. RANKING/LIMIT: if the intent asks for 'top N' or 'bottom N', the "
        "exact ORDER BY column and direction, and the LIMIT value.\n\n"
        "Then generate a Gold STTM CSV with the same 8 required columns "
        "(source_schema, source_table, source_column, target_schema, "
        "target_table, target_column, transformation_type, transformation_logic) "
        "encoding those four decisions as concrete rows. Use these "
        "transformation_type values:\n"
        "- join: transformation_logic = 'join_left:<real_fact_table>:<real_dim_table>:<real_key_column>'\n"
        "- time_filter: transformation_logic = the exact filter condition on the "
        "real date column (e.g. 'transaction_date >= last 4 quarters from MAX(transaction_date)')\n"
        "- group_by: transformation_logic = the real column(s) to group by\n"
        "- aggregate: transformation_logic = the real aggregate expression, e.g. "
        "'SUM(quantity * unit_price) AS total_revenue'\n"
        "- rank_limit: transformation_logic = e.g. 'ORDER BY total_revenue DESC LIMIT 5'\n\n"
        "If the intent has a time window or a top-N ranking, you MUST include a "
        "time_filter row and/or a rank_limit row -- do not silently drop them.\n"
        "Respond with a ```csv fenced code block containing ONLY the CSV."
    )
 
    # This is the hardest reasoning step in the pipeline -- translating an
    # arbitrary business question into the right join/time-filter/ranking
    # logic. "medium" effort with a moderate token ceiling balances
    # reasoning quality against the account's rate-limit budget (see
    # core/llm.py -- "high" effort combined with a large max_tokens can
    # exceed a lower-tier account's tokens-per-minute limit on its own).
    llm_text = call_llm(prompt, max_tokens=5500, reasoning_effort="medium")
    if not llm_text:
        raise RuntimeError("LLM unavailable -- cannot generate Gold STTM.")
 
    csv_text = _extract_csv(llm_text)
    out_path = STTM_DIR / f"sttm_gold_{run_id[:8]}.csv"
    _validate_and_save(csv_text, out_path, raw_llm_text=llm_text)
    audit.log("sttm_generator", "gold_sttm_generated", output_path=str(out_path))
    return str(out_path)