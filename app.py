"""
app.py — AI Hallucination-Aware RAG
Simple Streamlit UI — API key is set in config.py
"""

import os
import tempfile
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Hallucination-Aware RAG",
    page_icon="🔍",
    layout="centered",
)

st.markdown("""
<style>
    .block-container { max-width: 860px; }
    .answer-box {
        background: #f0f4ff;
        border-left: 4px solid #5c6bc0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        white-space: pre-wrap;
        margin-bottom: 1rem;
        color: #1a1a1a !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .fix-box {
        background: #e8f5e9;
        border-left: 4px solid #43a047;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        white-space: pre-wrap;
        margin-bottom: 1rem;
        color: #1a1a1a !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    h1 { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for key, val in [("sources", []), ("result", None)]:
    if key not in st.session_state:
        st.session_state[key] = val


# ── Initialize RAG once (cached so it only loads models once) ─────────────────
@st.cache_resource(show_spinner="Loading AI models — first run takes ~2 minutes...")
def load_rag():
    from src.rag_pipeline import HallucinationAwareRAG
    return HallucinationAwareRAG()


# ── Gauge chart ────────────────────────────────────────────────────────────────
def gauge(score: float):
    color = "#43a047" if score < 20 else "#fb8c00" if score < 50 else "#e53935"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 44, "color": color}},
        title={"text": "Hallucination Score", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 20],   "color": "#e8f5e9"},
                {"range": [20, 50],  "color": "#fff3e0"},
                {"range": [50, 100], "color": "#fce4ec"},
            ],
        }
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=0, l=10, r=10))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("🔍 Hallucination-Aware RAG")
st.caption("Upload documents → Ask a question → See what's real vs. hallucinated")
st.divider()

# Load RAG pipeline (cached)
rag = load_rag()

# ── Source upload section ──────────────────────────────────────────────────────
st.subheader("📂 Step 1 — Add Your Sources")

input_type = st.radio("Source type", ["📄 File", "🌐 URL"], horizontal=True)

if input_type == "📄 File":
    files = st.file_uploader(
        "Upload PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if st.button("➕ Add Files", type="primary"):
        if not files:
            st.warning("Please select at least one file.")
        else:
            for f in files:
                ext = "." + f.name.rsplit(".", 1)[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                with st.spinner(f"Processing {f.name}…"):
                    try:
                        n = rag.add_source(tmp_path)
                        st.session_state.sources.append(f.name)
                        st.success(f"✅ {f.name} — {n} chunks indexed")
                    except Exception as e:
                        st.error(f"❌ {f.name}: {e}")
                    finally:
                        os.unlink(tmp_path)
else:
    url = st.text_input("Paste a URL", placeholder="https://en.wikipedia.org/wiki/...")
    if st.button("➕ Add URL", type="primary"):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Fetching page…"):
                try:
                    n = rag.add_source(url.strip())
                    st.session_state.sources.append(url.strip())
                    st.success(f"✅ URL added — {n} chunks indexed")
                except Exception as e:
                    st.error(f"❌ {e}")

# Show loaded sources
if st.session_state.sources:
    with st.expander(f"📚 {len(st.session_state.sources)} source(s) loaded"):
        for s in st.session_state.sources:
            st.markdown(f"- `{s}`")
        if st.button("🗑️ Clear all sources"):
            rag.clear_sources()
            st.session_state.sources = []
            st.session_state.result = None
            st.rerun()

st.divider()

# ── Question section ───────────────────────────────────────────────────────────
st.subheader("❓ Step 2 — Ask a Question")

if not st.session_state.sources:
    st.info("Add at least one source above before asking a question.")
    st.stop()

question = st.text_input(
    "Your question",
    placeholder="e.g. What is the main conclusion of this document?",
    label_visibility="collapsed"
)

if st.button("🔍 Analyze", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving → Answering → Verifying… (30–60 seconds)"):
            try:
                st.session_state.result = rag.query(question.strip())
            except Exception as e:
                st.error(f"Error: {e}")

# ── Results ────────────────────────────────────────────────────────────────────
r = st.session_state.result
if not r:
    st.stop()

if "error" in r:
    st.error(r["error"])
    st.stop()

st.divider()
st.subheader("📊 Results")

# Gauge + metrics
left, right = st.columns([1, 1])
with left:
    st.plotly_chart(gauge(r["hallucination_score"]), use_container_width=True)
with right:
    score = r["hallucination_score"]
    st.metric("Claims Checked", len(r["claims"]))
    st.metric("✅ Supported",   sum(1 for c in r["claims"] if c["verdict"] == "supported"))
    st.metric("⚠️ / ❌ Issues", sum(1 for c in r["claims"] if c["verdict"] != "supported"))

    if score == 0:
        st.success("Fully grounded — no hallucinations detected!")
    elif score < 20:
        st.warning(f"Low hallucination ({score}%) — mostly reliable.")
    elif score < 50:
        st.warning(f"Moderate hallucination ({score}%) — check corrected answer.")
    else:
        st.error(f"High hallucination ({score}%) — original answer is unreliable.")

st.divider()

# Original answer
st.subheader("📝 Original Answer")
st.markdown(f'<div class="answer-box">{r["answer"]}</div>', unsafe_allow_html=True)

# Corrected answer (only shown if hallucinations found)
if r["corrected_answer"]:
    st.subheader("✅ Corrected Answer")
    st.markdown(f'<div class="fix-box">{r["corrected_answer"]}</div>', unsafe_allow_html=True)

st.divider()

# Claim breakdown
st.subheader("🔬 Claim-by-Claim Breakdown")
st.caption(r["check_summary"])

for i, claim in enumerate(r["claims"], 1):
    v = claim["verdict"]
    icon = "✅" if v == "supported" else ("⚠️" if v == "unsupported" else "❌")
    with st.expander(f"{icon} Claim {i}: {claim['claim'][:80]}{'…' if len(claim['claim']) > 80 else ''}"):
        st.write(f"**Verdict:** `{v.upper()}` — {claim['confidence']}% confidence")
        c1, c2, c3 = st.columns(3)
        for col, (lbl, val) in zip([c1, c2, c3], claim["scores"].items()):
            col.metric(lbl.capitalize(), f"{val}%")

# Source chunks
with st.expander("📄 View source chunks used to answer"):
    st.text(r["context_used"])
st.caption(f"Sources: {', '.join(r['sources_referenced'])}")