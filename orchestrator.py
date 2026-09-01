"""
Interactive pipeline orchestrator with HITL approval gates.
"""
import argparse
import os
import sys
import uuid
 
import pandas as pd
from tabulate import tabulate
 
from agents import profiler, sttm_generator, bronze_agent, silver_agent, gold_agent, reporter
from core.state import PipelineState
from core.config import ensure_dirs
 
 
def banner(text: str) -> None:
    print(f"\n\033[1;34m== {text} ==\033[0m\n")
 
 
def display_sttm(sttm_path: str, layer_name: str) -> None:
    df = pd.read_csv(sttm_path)
    banner(f"{layer_name} STTM - {len(df)} rules")
    print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))
    print(f" File: {sttm_path}")
 
 
def hitl_gate(layer_name: str, sttm_path: str) -> bool:
    display_sttm(sttm_path, layer_name)
    while True:
        prompt = f"\n[{layer_name} STTM] Approve? [y]es / [e]dit then re-review / [n]o abort > "
        choice = input(prompt).strip().lower()
        if choice == "y":
            print(f" \u2713 {layer_name} STTM approved.")
            return True
        elif choice == "e":
            editor = os.environ.get("EDITOR", "nano")
            os.system(f"{editor} {sttm_path}")
            display_sttm(sttm_path, layer_name)
            continue
        elif choice == "n":
            print(f" \u2717 Pipeline aborted at {layer_name} gate.")
            return False
        else:
            print(" Please enter y, e, or n.")
 
 
def run_pipeline(files: list[str], intent: str) -> PipelineState:
    ensure_dirs()
    run_id = str(uuid.uuid4())
    state = PipelineState(run_id=run_id, input_files=files, business_intent=intent)
 
    banner("Phase 1 - Profiling + Bronze STTM")
    state.profile_path = profiler.profile(files, run_id)
    print(f" -> Profile: {state.profile_path}")
    state.bronze_sttm_path = sttm_generator.generate_bronze_sttm(state.profile_path, intent, run_id)
    if not hitl_gate("Bronze", state.bronze_sttm_path):
        return state
 
    banner("Phase 2 - Bronze Agent + Silver STTM")
    state.bronze_paths = bronze_agent.run(files, state.bronze_sttm_path, run_id)
    print(f" -> Bronze parquets: {state.bronze_paths}")
    state.silver_sttm_path = sttm_generator.generate_silver_sttm(
        state.bronze_paths, state.bronze_sttm_path, intent, run_id
    )
    if not hitl_gate("Silver", state.silver_sttm_path):
        return state
 
    banner("Phase 3 - Silver Agent + Gold STTM")
    state.silver_paths = silver_agent.run(state.bronze_paths, state.silver_sttm_path, run_id)
    print(f" -> Silver parquets: {state.silver_paths}")
    state.gold_sttm_path = sttm_generator.generate_gold_sttm(
        state.silver_paths, state.silver_sttm_path, intent, run_id
    )
    if not hitl_gate("Gold", state.gold_sttm_path):
        return state
 
    banner("Phase 4 - Gold Agent + Report")
    state.gold_paths = gold_agent.run(state.silver_paths, state.gold_sttm_path, intent, run_id)
    print(f" -> Gold parquets: {state.gold_paths}")
    state.report_path = reporter.generate_report(state.gold_paths, intent, run_id)
    print(f" -> Report: {state.report_path}")
 
    banner("Pipeline Complete")
    print(f" Run ID : {run_id}")
    print(f" Report : {state.report_path}")
    print(f" Audit log : data/audit_logs/")
    print(f"\n Open report: python -m http.server 8080 --directory data/reports")
 
    return state
 
 
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", help="Input CSV file paths")
    parser.add_argument("--intent", type=str, help="Business question / intent")
    args = parser.parse_args()
 
    if args.files is None:
        files_input = input("Enter CSV file paths (space-separated): ")
        files = files_input.strip().split()
    else:
        files = args.files
 
    if args.intent is None:
        intent = input("Enter business intent / question: ").strip()
    else:
        intent = args.intent
 
    if not files or not intent:
        print("Error: both --files and --intent are required.")
        sys.exit(1)
 
    run_pipeline(files, intent)
 
 
if __name__ == "__main__":
    main()
