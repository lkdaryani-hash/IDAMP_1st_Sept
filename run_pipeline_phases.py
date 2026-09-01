import uuid
from core.config import ensure_dirs
from agents import profiler, sttm_generator, bronze_agent, silver_agent, gold_agent, reporter

ensure_dirs()
files = ['raw_files/sales_data.csv','raw_files/products.csv','raw_files/stores.csv']
intent = "Top 5 revenue-generating products in the last 4 quarters?"
run_id = str(uuid.uuid4())
print('RUN_ID:', run_id)

# Phase 1: profiling
try:
    profile_path = profiler.profile(files, run_id)
    print('Profile:', profile_path)
except Exception as e:
    print('Profile failed:', e)
    raise

# Phase 1.5: generate bronze STTM
try:
    bronze_sttm = sttm_generator.generate_bronze_sttm(profile_path, intent, run_id)
    print('Bronze STTM:', bronze_sttm)
except Exception as e:
    print('Bronze STTM generation failed:', e)
    raise

# Phase 2: bronze agent
try:
    bronze_paths = bronze_agent.run(files, bronze_sttm, run_id)
    print('Bronze parquet paths:', bronze_paths)
except Exception as e:
    print('Bronze agent failed:', e)
    raise

# Phase 2.5: generate silver STTM
try:
    silver_sttm = sttm_generator.generate_silver_sttm(bronze_paths, bronze_sttm, intent, run_id)
    print('Silver STTM:', silver_sttm)
except Exception as e:
    print('Silver STTM generation failed:', e)
    raise

# Phase 3: silver agent
try:
    silver_paths = silver_agent.run(bronze_paths, silver_sttm, run_id)
    print('Silver parquet paths:', silver_paths)
except Exception as e:
    print('Silver agent failed:', e)
    raise

# Phase 3.5: generate gold STTM
try:
    gold_sttm = sttm_generator.generate_gold_sttm(silver_paths, silver_sttm, intent, run_id)
    print('Gold STTM:', gold_sttm)
except Exception as e:
    print('Gold STTM generation failed:', e)
    raise

# Phase 4: gold agent
try:
    gold_paths = gold_agent.run(silver_paths, gold_sttm, intent, run_id)
    print('Gold parquet paths:', gold_paths)
except Exception as e:
    print('Gold agent failed:', e)
    raise

# Final: reporter
try:
    report_path = reporter.generate_report(gold_paths, intent, run_id)
    print('Report path:', report_path)
except Exception as e:
    print('Report generation failed:', e)
    raise

print('Dry run complete')
