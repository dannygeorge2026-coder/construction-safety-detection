import streamlit as st
import cv2
import csv
import tempfile
import os
from pathlib import Path
from ultralytics import YOLO

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Safety Monitor",
    page_icon="🦺",
    layout="wide",
)

# ── Minimal dark theme override ───────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp { background: #0d1117; color: #e6edf3; }

  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    text-align: center;
  }
  .metric-card .label { font-size: 0.75rem; color: #8b949e; letter-spacing: 0.08em; text-transform: uppercase; }
  .metric-card .value { font-size: 2.2rem; font-weight: 700; margin-top: 0.2rem; }
  .metric-card .value.danger  { color: #f85149; }
  .metric-card .value.warning { color: #d29922; }
  .metric-card .value.ok      { color: #3fb950; }

  .section-header {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8b949e;
    margin: 1.6rem 0 0.6rem;
  }

  .stButton > button {
    background: #238636;
    color: #fff;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
    transition: background 0.15s;
  }
  .stButton > button:hover { background: #2ea043; }

  .stProgress > div > div { background: #238636; }

  /* hide default metric styling */
  [data-testid="stMetricValue"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🦺 Construction Safety Monitor")
st.markdown('<p style="color:#8b949e;margin-top:-0.5rem;">Upload a site video · detect PPE violations · export report</p>', unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model")
    model_path = st.text_input("Weights path", value="models/best.pt")

    st.markdown("### Detection")
    conf_thresh = st.slider("Confidence threshold", 0.1, 0.95, 0.5, 0.05)
    show_live   = st.checkbox("Show frame preview", value=True)
    save_video  = st.checkbox("Save annotated video", value=True)

VIOLATION_CLASSES = ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Input Video</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Drop a video file here", type=["mp4", "avi", "mov", "mkv"])

if not uploaded:
    st.info("Upload a video above to start analysis.")
    st.stop()

# ── Run analysis ──────────────────────────────────────────────────────────────
if st.button("▶  Run Analysis"):

    # Load model
    if not Path(model_path).exists():
        st.error(f"Model not found at `{model_path}`. Check the path in the sidebar.")
        st.stop()

    with st.spinner("Loading model…"):
        model = YOLO(model_path)

    # Write upload to temp file
    tmp_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_in.write(uploaded.read())
    tmp_in.close()

    cap = cv2.VideoCapture(tmp_in.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    # Output video temp file
    tmp_out_path = None
    out = None
    if save_video:
        tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_out_path = tmp_out.name
        tmp_out.close()
        out = cv2.VideoWriter(tmp_out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    violations  = {c: 0 for c in VIOLATION_CLASSES}
    frame_number = 0

    st.markdown('<div class="section-header">Processing</div>', unsafe_allow_html=True)
    progress_bar  = st.progress(0)
    status_text   = st.empty()
    preview_slot  = st.empty() if show_live else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        results = model(frame, conf=conf_thresh, verbose=False)
        annotated = results[0].plot()

        current_violations = 0
        for box in results[0].boxes:
            cls  = int(box.cls[0])
            name = model.names[cls]
            if name in violations:
                violations[name] += 1
                current_violations += 1

        cv2.putText(annotated, f"Violations: {current_violations}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if out:
            out.write(annotated)

        pct = frame_number / max(total_frames, 1)
        progress_bar.progress(min(pct, 1.0))
        status_text.markdown(f'<span style="color:#8b949e;font-size:0.85rem;">Frame {frame_number} / {total_frames} &nbsp;·&nbsp; violations this frame: {current_violations}</span>', unsafe_allow_html=True)

        # Preview every 10th frame
        if show_live and frame_number % 10 == 0:
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            preview_slot.image(rgb, channels="RGB", use_container_width=True)

    cap.release()
    if out:
        out.release()
    os.unlink(tmp_in.name)

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)

    total_v = sum(violations.values())

    cols = st.columns(4)
    def metric_card(col, label, value, tone="ok"):
        col.markdown(f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value {tone}">{value}</div>
        </div>""", unsafe_allow_html=True)

    metric_card(cols[0], "Frames processed", frame_number, "ok")
    metric_card(cols[1], "Total violations",  total_v,      "danger" if total_v > 0 else "ok")
    metric_card(cols[2], "No hardhat",  violations["NO-Hardhat"],      "danger" if violations["NO-Hardhat"] else "ok")
    metric_card(cols[3], "No vest",     violations["NO-Safety Vest"],  "danger" if violations["NO-Safety Vest"] else "ok")

    # Breakdown table
    st.markdown('<div class="section-header">Violation Breakdown</div>', unsafe_allow_html=True)
    for vtype, count in violations.items():
        bar_pct = count / max(total_v, 1)
        color   = "#f85149" if count > 0 else "#3fb950"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.5rem;">
          <span style="width:160px;font-size:0.85rem;color:#e6edf3;">{vtype}</span>
          <div style="flex:1;background:#21262d;border-radius:4px;height:10px;">
            <div style="width:{bar_pct*100:.1f}%;background:{color};height:10px;border-radius:4px;"></div>
          </div>
          <span style="width:40px;text-align:right;font-family:'JetBrains Mono';font-size:0.9rem;color:{color};">{count}</span>
        </div>""", unsafe_allow_html=True)

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Downloads</div>', unsafe_allow_html=True)
    dl_cols = st.columns(2)

    # CSV report
    import io
    csv_buf = io.StringIO()
    writer  = csv.writer(csv_buf)
    writer.writerow(["Violation Type", "Count"])
    for vtype, count in violations.items():
        writer.writerow([vtype, count])

    dl_cols[0].download_button(
        "⬇  Download CSV Report",
        data     = csv_buf.getvalue().encode(),
        file_name = "violation_report.csv",
        mime     = "text/csv",
    )

    # Annotated video
    if tmp_out_path and os.path.exists(tmp_out_path):
        with open(tmp_out_path, "rb") as f:
            dl_cols[1].download_button(
                "⬇  Download Annotated Video",
                data      = f.read(),
                file_name = "output_annotated.mp4",
                mime      = "video/mp4",
            )
        os.unlink(tmp_out_path)

    st.success("✅ Analysis complete.")