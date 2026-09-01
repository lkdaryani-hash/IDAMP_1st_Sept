"""
Gold Agent.
 
Gold is the analytics layer: joins Silver tables and applies
GROUP BY + aggregations to answer the business intent. LLM (Groq)
generates the DuckDB SQL from the Gold STTM + schemas.
"""
import re
from pathlib import Path
 
import duckdb
import pandas as pd
 
from core.config import GOLD_DIR
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
 
 
def _fix_table_names(sql: str, real_table_names: list[str]) -> str:
    """
    The model is given exact table names like 'sales_data_silver_f09f789f'
    (base name + run-id suffix), but sometimes drops the suffix and writes
    the shorter, more "natural" looking 'sales_data_silver' instead. DuckDB
    then fails with a clear "table does not exist" error that even names
    the correct table -- rather than re-calling the LLM (slower, costs
    tokens, and isn't guaranteed to get it right either), just do the
    substitution directly: for each real table name, if the SQL references
    its bare prefix (with the run-id suffix stripped) as a standalone word,
    replace it with the real, exact table name.
    """
    for real_name in real_table_names:
        stripped = re.sub(r"_[0-9a-f]{8}$", "", real_name)
        if stripped == real_name:
            continue  # this table name has no run-id suffix to strip
        sql = re.sub(rf"\b{re.escape(stripped)}\b", real_name, sql)
    return sql
 
 
def run(
    silver_paths: list[str],
    sttm_path: str,
    business_intent: str,
    run_id: str,
) -> list[str]:
    audit = AuditLogger(run_id)
    sttm_df = pd.read_csv(sttm_path)
    conn = duckdb.connect()
    table_registry: dict[str, str] = {}
 
    schema_lines = []
    for fp in silver_paths:
        raw_stem = Path(fp).stem
        # sanitize a safe DuckDB table name (no spaces or special chars)
        clean_name = re.sub(r"\W+", "_", raw_stem)
        safe_path = str(fp).replace("\\", "\\\\")
        conn.execute(
            f"CREATE OR REPLACE TABLE {clean_name} AS SELECT * FROM read_parquet('{safe_path}')"
        )
        table_registry[clean_name] = fp
        row_count = conn.execute(f"SELECT COUNT(*) FROM {clean_name}").fetchone()[0]
        columns = conn.execute(f"DESCRIBE {clean_name}").df()["column_name"].tolist()
        schema_lines.append(f"Table {clean_name}: {row_count} rows, columns={columns}")
 
    prompt = (
        f"Business intent: {business_intent}\n\n"
        f"Table schemas:\n" + "\n".join(schema_lines) + "\n\n"
        f"Gold STTM rules:\n{sttm_df.to_string(index=False)}\n\n"
        "Instructions:\n"
        "- Parse join rules 'join_left:A:B:key' as 'A LEFT JOIN B ON A.key = B.key', "
        "using the REAL business key column named in the rule (e.g. product_id) "
        "on BOTH sides. NEVER join on a pk_*_silver_id column -- that is an "
        "auto-generated row number added separately in Python, unrelated to "
        "any business key, and joining on it produces meaningless matches.\n"
        "- CRITICAL: before finalizing any join, check the exact column list "
        "shown for BOTH tables above. If the intended key (e.g. product_id) "
        "is listed for one table but NOT the other, do NOT join a column "
        "that exists on one side against a *different kind* of column on "
        "the other side (e.g. never write "
        "'ON a.product_id = b.product_name' -- an ID will never equal a "
        "name, so that join silently matches nothing). Instead, pick "
        "whichever single column name is actually present in BOTH tables' "
        "column lists and join on that same column on both sides.\n"
        "- Parse group_by and aggregate rules\n"
        "- IMPORTANT: any *_date column in these Silver tables is stored as "
        "VARCHAR (e.g. '2024-01-15'), not a native DATE/TIMESTAMP type. "
        "Comparing or doing arithmetic on a VARCHAR date against a computed "
        "DATE value can silently match zero rows instead of erroring. "
        "ALWAYS wrap any date column in CAST(col AS DATE) or col::DATE "
        "before using it in date arithmetic or a WHERE comparison.\n"
        "- IMPORTANT: DuckDB does NOT have a DATEADD() function -- using it "
        "will fail with a Catalog Error. For date arithmetic, subtract or "
        "add an INTERVAL directly, e.g.: "
        "some_date_expr - INTERVAL '12 months'  (for 4 quarters back, since "
        "1 quarter = 3 months, always convert quarters to months this way).\n"
        "- Parse any time_filter rule as a WHERE condition on the real date "
        "column it references (cast per the rule above, using INTERVAL "
        "arithmetic per the rule above -- never DATEADD). If the rule "
        "anchors on MAX(date_column), you MUST use "
        "(SELECT MAX(date_column::DATE) FROM the_real_fact_table) as the "
        "anchor -- do NOT substitute CURRENT_DATE() or CURRENT_DATE, since "
        "the data's own dates may be far from today's real-world date and "
        "that substitution can silently filter out all rows.\n"
        "- Parse any rank_limit rule as the ORDER BY + LIMIT clause of the "
        "final SELECT -- do not omit this if present\n"
        "- Generate ONE DuckDB SELECT with JOINs, GROUP BY, aggregates, the "
        "time filter (if any), and ORDER BY/LIMIT (if any)\n"
        "- Do NOT add pk_gold_id (added afterward in Python)\n"
        "- Order by the main aggregate DESC\n"
        "Respond with a ```sql fenced code block containing ONLY the SQL."
    )
 
    df = None
    sql_used = None
    sql_error = None
 
    # Pick the largest table as the fallback default -- if the LLM's SQL
    # fails, dumping the fact table (most rows) is far more useful than an
    # arbitrary dimension table, even though it still won't be aggregated.
    row_counts = {
        name: conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        for name in table_registry
    }
    fallback_table = max(row_counts, key=row_counts.get)
 
    llm_text = ""
    try:
        # Same account-tier constraint as the Gold STTM step: on_demand tier
        # caps requests at 8000 tokens/minute total (prompt + max_tokens),
        # so "high" effort with a large ceiling can trip a 413 rate-limit
        # error. "medium" fits safely while still reasoning through the
        # join/date-cast/ranking logic meaningfully better than "low".
        llm_text = call_llm(prompt, max_tokens=3500, reasoning_effort="medium")
        sql = _extract_sql(llm_text)
        if sql:
            sql = _fix_table_names(sql, list(table_registry.keys()))
            df = conn.execute(sql).df()
            sql_used = sql
        else:
            sql_error = "No SQL could be extracted from the LLM response."
    except Exception as e:
        sql_error = str(e)
        df = None
 
    if df is None:
        # Save exactly what went wrong -- the SQL that was attempted (if any)
        # and the real DuckDB/parsing error -- so this is diagnosable instead
        # of silently falling back every time.
        debug_path = GOLD_DIR / f"gold_sql_failure_{run_id[:8]}.txt"
        debug_path.write_text(
            f"ERROR: {sql_error}\n\n"
            f"--- RAW LLM RESPONSE ---\n{llm_text}\n",
            encoding="utf-8",
        )
        audit.log(
            "gold_agent",
            "sql_generation_failed",
            error=sql_error,
            debug_file=str(debug_path),
        )
        df = pd.read_parquet(table_registry[fallback_table])
        sql_used = f"FALLBACK: pd.read_parquet({fallback_table}) -- see {debug_path.name} for why"
 
    elif len(df) == 0:
        # The SQL ran without error and has the right shape, but matched
        # zero rows -- almost always a silent join or WHERE-filter mismatch.
        # Flag it loudly here with the exact SQL that produced nothing,
        # rather than let it silently reach the final report as a blank
        # table.
        debug_path = GOLD_DIR / f"gold_sql_zero_rows_{run_id[:8]}.txt"
        debug_path.write_text(
            f"SQL executed successfully but returned 0 rows.\n"
            f"This usually means a join key or WHERE/date filter didn't "
            f"actually match anything -- check column value formats on "
            f"both sides of the join and any date casts/anchors.\n\n"
            f"--- SQL EXECUTED ---\n{sql_used}\n",
            encoding="utf-8",
        )
        audit.log(
            "gold_agent",
            "sql_returned_zero_rows",
            sql=sql_used,
            debug_file=str(debug_path),
        )
 
    df.insert(0, "pk_gold_id", range(1, len(df) + 1))
 
    out_path = GOLD_DIR / f"sales_analytics_gold_{run_id[:8]}.parquet"
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
    df.to_parquet(out_path, index=False)
 
    audit.log(
        "gold_agent",
        "gold_table_generated",
        output_path=str(out_path),
        row_count=len(df),
        sql=sql_used,
    )
 
    return [str(out_path)]