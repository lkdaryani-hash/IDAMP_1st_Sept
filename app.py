"""
Streamlit web UI wrapping the entire IDAMP pipeline.
"""
import json
import pathlib
import uuid
 
import pandas as pd
import streamlit as st
 
from core.config import ensure_dirs, STTM_DIR, REPORTS_DIR, AUDIT_DIR, UPLOADS_DIR
from core.state import PipelineState
from agents import profiler, sttm_generator, bronze_agent, silver_agent, gold_agent, reporter
 
st.set_page_config(
    page_title="IDAMP - Agentic Medallion Pipeline",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
 
defaults = {
    "stage": "input",
    "state": None,
    "run_id": None,
    "input_files": [],
    "intent": "",
    "log_messages": [],
    "report_html": None,
    "edited_sttm": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
 
STAGE_STEPS = [
    ("input", "Upload & Intent"),
    ("running_phase1", "Profile + Bronze STTM"),
    ("bronze_hitl", "[HITL] Review Bronze STTM"),
    ("running_phase2", "Bronze Agent"),
    ("silver_hitl", "[HITL] Review Silver STTM"),
    ("running_phase3", "Silver Agent"),
    ("gold_hitl", "[HITL] Review Gold STTM"),
    ("running_phase4", "Gold Agent + Report"),
]
 
 
def render_sidebar():
    st.sidebar.title("Pipeline Status")
    current = st.session_state["stage"]
    order = [s[0] for s in STAGE_STEPS] + ["complete"]
    current_idx = order.index(current) if current in order else -1
 
    for idx, (key, label) in enumerate(STAGE_STEPS):
        if current == "aborted":
            icon = "❌" if idx <= current_idx else "⚪"
        elif idx < current_idx or current == "complete":
            icon = "✅"
        elif idx == current_idx:
            icon = "🔵"
        else:
            icon = "⚪"
        st.sidebar.write(f"{icon} {label}")
 
    st.sidebar.divider()
    if st.session_state["run_id"]:
        st.sidebar.code(st.session_state["run_id"][:8])
    if st.session_state["intent"]:
        st.sidebar.caption(st.session_state["intent"])
    st.sidebar.caption(f"Log messages: {len(st.session_state['log_messages'])}")
 
 
render_sidebar()
 
# ---------------- STAGE: input ----------------
if st.session_state["stage"] == "input":
    st.title("🥇 Intent-Driven Agentic Medallion Pipeline")
    st.markdown(
        "Upload your raw CSV files and enter a business question. "
        "The pipeline will profile, clean, aggregate, and generate an HTML analytics report."
    )
 
    uploaded_files = st.file_uploader(
        "Upload raw CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload all source CSV files (e.g. sales_data.csv, products.csv, stores.csv)",
    )
 
    saved_paths = []
    if uploaded_files:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        for file in uploaded_files:
            dest = UPLOADS_DIR / file.name
            dest.write_bytes(file.read())
            saved_paths.append(str(dest))
        st.session_state["input_files"] = saved_paths
 
    intent = st.text_area(
        "Business Question / Intent",
        placeholder="Which category had the highest Q1 2024 revenue, and how does it break down by region?",
        height=80,
    )
 
    with st.expander("Example business questions"):
        st.write("- Which customer segment had the highest total spending?")
        st.write("- How does revenue break down by region and payment method?")
        st.write("- Which products have the highest return rate?")
        st.write("- What is the average order value per store?")
        st.write("- Top 5 revenue-generating products in the last 4 quarters?")
 
    if st.button("🚀 Start Pipeline", type="primary", disabled=(not uploaded_files or not intent)):
        ensure_dirs()
        st.session_state["run_id"] = str(uuid.uuid4())
        st.session_state["intent"] = intent.strip()
        st.session_state["stage"] = "running_phase1"
        st.rerun()
 
# ---------------- STAGE: running_phase1 ----------------
elif st.session_state["stage"] == "running_phase1":
    with st.spinner("Phase 1 - Profiling data and generating Bronze STTM..."):
        run_id = st.session_state["run_id"]
        files = st.session_state["input_files"]
        intent = st.session_state["intent"]
        pipeline_state = PipelineState(run_id=run_id, input_files=files, business_intent=intent)
        pipeline_state.profile_path = profiler.profile(files, run_id)
        pipeline_state.bronze_sttm_path = sttm_generator.generate_bronze_sttm(
            pipeline_state.profile_path, intent, run_id
        )
        st.session_state["state"] = pipeline_state
        st.session_state["stage"] = "bronze_hitl"
        st.rerun()
 
# ---------------- STAGE: bronze_hitl ----------------
elif st.session_state["stage"] == "bronze_hitl":
    st.subheader("🔶 Review Bronze STTM")
    st.info(
        "The LLM has generated the Bronze transformation rules. "
        "Review them below. You can edit any cell before approving."
    )
 
    sttm_path = st.session_state["state"].bronze_sttm_path
    df = pd.read_csv(sttm_path)
    edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic", key="bronze_editor")
 
    col1, col2, col3 = st.columns(3)
    col1.metric("Total rules", len(df))
    col2.metric("type_cast rules", len(df[df.transformation_type == "type_cast"]))
    col3.metric("metadata_inject rules", len(df[df.transformation_type == "metadata_inject"]))
 
    b1, b2, b3 = st.columns(3)
    if b1.button("✅ Approve & Continue", type="primary"):
        edited_df.to_csv(sttm_path, index=False)
        st.session_state["stage"] = "running_phase2"
        st.rerun()
    if b2.button("🔄 Regenerate Bronze STTM"):
        with st.spinner("Regenerating..."):
            new_path = sttm_generator.generate_bronze_sttm(
                st.session_state["state"].profile_path,
                st.session_state["intent"],
                st.session_state["run_id"],
            )
            st.session_state["state"].bronze_sttm_path = new_path
            st.rerun()
    if b3.button("❌ Abort Pipeline"):
        st.session_state["stage"] = "aborted"
        st.rerun()
 
# ---------------- STAGE: running_phase2 ----------------
elif st.session_state["stage"] == "running_phase2":
    with st.spinner("Phase 2 - Running Bronze Agent + generating Silver STTM..."):
        state = st.session_state["state"]
        state.bronze_paths = bronze_agent.run(state.input_files, state.bronze_sttm_path, state.run_id)
        state.silver_sttm_path = sttm_generator.generate_silver_sttm(
            state.bronze_paths, state.bronze_sttm_path, st.session_state["intent"], state.run_id
        )
        st.session_state["state"] = state
        st.session_state["stage"] = "silver_hitl"
        st.rerun()
 
# ---------------- STAGE: silver_hitl ----------------
elif st.session_state["stage"] == "silver_hitl":
    st.subheader("⬜ Review Silver STTM")
    st.info(
        "Silver rules cleanse and standardize the Bronze data. "
        "Review normalization rules carefully - especially normalize_category and standardize_date."
    )
 
    sttm_path = st.session_state["state"].silver_sttm_path
    df = pd.read_csv(sttm_path)
    edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic", key="silver_editor")
 
    col1, col2, col3 = st.columns(3)
    col1.metric("null_handler rules", len(df[df.transformation_type == "null_handler"]))
    col2.metric(
        "normalize rules",
        len(df[df.transformation_type.isin(["normalize_category", "normalize_segment"])]),
    )
    col3.metric("dedup rules", len(df[df.transformation_type == "dedup"]))
 
    b1, b2, b3 = st.columns(3)
    if b1.button("✅ Approve & Continue", type="primary"):
        edited_df.to_csv(sttm_path, index=False)
        st.session_state["stage"] = "running_phase3"
        st.rerun()
    if b2.button("🔄 Regenerate Silver STTM"):
        with st.spinner("Regenerating..."):
            state = st.session_state["state"]
            new_path = sttm_generator.generate_silver_sttm(
                state.bronze_paths, state.bronze_sttm_path, st.session_state["intent"], state.run_id
            )
            state.silver_sttm_path = new_path
            st.rerun()
    if b3.button("❌ Abort Pipeline"):
        st.session_state["stage"] = "aborted"
        st.rerun()
 
# ---------------- STAGE: running_phase3 ----------------
elif st.session_state["stage"] == "running_phase3":
    with st.spinner("Phase 3 - Running Silver Agent + generating Gold STTM..."):
        state = st.session_state["state"]
        state.silver_paths = silver_agent.run(state.bronze_paths, state.silver_sttm_path, state.run_id)
        state.gold_sttm_path = sttm_generator.generate_gold_sttm(
            state.silver_paths, state.silver_sttm_path, st.session_state["intent"], state.run_id
        )
        st.session_state["state"] = state
        st.session_state["stage"] = "gold_hitl"
        st.rerun()
 
# ---------------- STAGE: gold_hitl ----------------
elif st.session_state["stage"] == "gold_hitl":
    st.subheader("🥇 Review Gold STTM")
    st.info(
        "Gold rules define how tables are joined and aggregated to answer your "
        "business question. Verify the join keys, time filters, and ranking/limit "
        "rules actually reference real columns from your data before approving."
    )
 
    sttm_path = st.session_state["state"].gold_sttm_path
    df = pd.read_csv(sttm_path)
    edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic", key="gold_editor")
 
    col1, col2, col3 = st.columns(3)
    col1.metric("join rules", len(df[df.transformation_type == "join"]))
    col2.metric("time_filter / rank_limit", len(df[df.transformation_type.isin(["time_filter", "rank_limit"])]))
    col3.metric("aggregate rules", len(df[df.transformation_type == "aggregate"]))
 
    st.markdown("**Business intent being answered:**")
    st.info(st.session_state["intent"])
 
    b1, b2, b3 = st.columns(3)
    if b1.button("✅ Approve & Continue", type="primary"):
        edited_df.to_csv(sttm_path, index=False)
        st.session_state["stage"] = "running_phase4"
        st.rerun()
    if b2.button("🔄 Regenerate Gold STTM"):
        with st.spinner("Regenerating..."):
            state = st.session_state["state"]
            new_path = sttm_generator.generate_gold_sttm(
                state.silver_paths, state.silver_sttm_path, st.session_state["intent"], state.run_id
            )
            state.gold_sttm_path = new_path
            st.rerun()
    if b3.button("❌ Abort Pipeline"):
        st.session_state["stage"] = "aborted"
        st.rerun()
 
# ---------------- STAGE: running_phase4 ----------------
elif st.session_state["stage"] == "running_phase4":
    with st.spinner("Phase 4 - Running Gold Agent and generating your analytics report (final LLM call)..."):
        state = st.session_state["state"]
        state.gold_paths = gold_agent.run(
            state.silver_paths, state.gold_sttm_path, st.session_state["intent"], state.run_id
        )
        state.report_path = reporter.generate_report(
            state.gold_paths, st.session_state["intent"], state.run_id
        )
        st.session_state["state"] = state
        st.session_state["report_html"] = pathlib.Path(state.report_path).read_text(encoding="utf-8")
        st.session_state["stage"] = "complete"
        st.rerun()
 
# ---------------- STAGE: complete ----------------
elif st.session_state["stage"] == "complete":
    st.success("✅ Pipeline complete!")
    st.subheader("📊 Analytics Report")
 
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Report", "🥇 Gold Data", "📋 All STTMs", "🪵 Audit Log"])
 
    with tab1:
        report_html = st.session_state["report_html"]
        st.components.v1.html(report_html, height=900, scrolling=True)
        st.download_button(
            label="⬇️ Download HTML Report",
            data=report_html,
            file_name=f"report_{st.session_state['run_id'][:8]}.html",
            mime="text/html",
        )
 
    with tab2:
        state = st.session_state["state"]
        for gp in state.gold_paths:
            df = pd.read_parquet(gp)
            st.dataframe(df, use_container_width=True)
            st.caption(f"Source: {gp} | {len(df)} rows, {len(df.columns)} columns")
 
    with tab3:
        s1, s2, s3 = st.tabs(["🔶 Bronze", "⬜ Silver", "🥇 Gold"])
        state = st.session_state["state"]
        with s1:
            if state.bronze_sttm_path:
                st.dataframe(pd.read_csv(state.bronze_sttm_path), use_container_width=True)
        with s2:
            if state.silver_sttm_path:
                st.dataframe(pd.read_csv(state.silver_sttm_path), use_container_width=True)
        with s3:
            if state.gold_sttm_path:
                st.dataframe(pd.read_csv(state.gold_sttm_path), use_container_width=True)
 
    with tab4:
        state = st.session_state["state"]
        audit_file = AUDIT_DIR / f"{state.run_id}.jsonl"
        if audit_file.exists():
            entries = [json.loads(l) for l in audit_file.read_text().splitlines() if l.strip()]
            st.metric("Total audit entries", len(entries))
            st.dataframe(pd.DataFrame(entries), use_container_width=True)
        else:
            st.info("No audit log found for this run.")
 
    st.divider()
    if st.button("🔁 Start a New Run"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
 
# ---------------- STAGE: aborted ----------------
elif st.session_state["stage"] == "aborted":
    st.error("❌ Pipeline aborted.")
    st.markdown("The pipeline was stopped at the HITL review stage. No report was generated.")
 
    state = st.session_state.get("state")
    if state:
        if state.bronze_paths:
            st.success(f"Bronze layer: {len(state.bronze_paths)} files written")
        if state.silver_paths:
            st.success(f"Silver layer: {len(state.silver_paths)} files written")
 
    if st.button("🔁 Start a New Run"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
