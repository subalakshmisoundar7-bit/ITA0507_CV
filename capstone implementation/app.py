import streamlit as st
import os
import json

from preprocessing1.preprocessing import preprocess_image
from segmentation2.segmentation import segment_defects
from classification3.classification import generate_report


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MetalVision AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #f4f7ff 0%,
        #eef5ff 45%,
        #f8f1ff 100%
    );
}

/* Main container */

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Header */

.hero {
    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed,
        #db2777
    );

    padding: 32px;
    border-radius: 24px;

    color: white;

    box-shadow:
        0 12px 30px rgba(67, 56, 202, 0.25);

    margin-bottom: 25px;
}

.hero-title {
    font-size: 40px;
    font-weight: 800;
    letter-spacing: 1px;
}

.hero-subtitle {
    font-size: 16px;
    opacity: 0.9;
    margin-top: 5px;
}

.online {
    background: rgba(255,255,255,0.18);
    padding: 10px 18px;
    border-radius: 30px;
    display: inline-block;
    font-weight: 600;
}

/* Section title */

.section-title {
    font-size: 22px;
    font-weight: 800;
    color: #172554;
    margin-top: 28px;
    margin-bottom: 15px;
}

/* Metric cards */

.metric-card {
    padding: 22px;
    border-radius: 18px;
    color: white;
    min-height: 120px;

    box-shadow:
        0 8px 22px rgba(0,0,0,0.10);
}

.metric-title {
    font-size: 13px;
    opacity: 0.9;
}

.metric-number {
    font-size: 34px;
    font-weight: 800;
    margin-top: 8px;
}

.blue {
    background: linear-gradient(
        135deg,
        #2563eb,
        #3b82f6
    );
}

.red {
    background: linear-gradient(
        135deg,
        #dc2626,
        #f97316
    );
}

.orange {
    background: linear-gradient(
        135deg,
        #ea580c,
        #f59e0b
    );
}

.green {
    background: linear-gradient(
        135deg,
        #059669,
        #22c55e
    );
}

/* Pipeline */

.pipeline-card {
    background: white;
    border-radius: 18px;
    padding: 22px;

    box-shadow:
        0 7px 20px rgba(37, 99, 235, 0.08);

    border: 1px solid #e2e8f0;
}

.pipeline-step {
    text-align: center;
}

.pipeline-number {
    width: 50px;
    height: 50px;

    margin: auto;

    border-radius: 50%;

    display: flex;
    align-items: center;
    justify-content: center;

    color: white;

    font-weight: 800;
    font-size: 15px;

    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );

    box-shadow:
        0 5px 15px rgba(79,70,229,0.25);
}

.pipeline-label {
    margin-top: 10px;
    font-size: 12px;
    font-weight: 700;
    color: #334155;
}

.pipeline-arrow {
    font-size: 25px;
    color: #94a3b8;
    text-align: center;
    padding-top: 10px;
}

/* Image cards */

.image-card {
    background: white;
    padding: 16px;
    border-radius: 18px;

    box-shadow:
        0 8px 25px rgba(15,23,42,0.08);

    border: 1px solid #e2e8f0;
}

.image-title {
    font-weight: 800;
    color: #334155;
    margin-bottom: 12px;
}

/* Result */

.result-card {
    background: white;
    border-radius: 15px;
    padding: 18px;

    margin-bottom: 10px;

    border-left: 6px solid #7c3aed;

    box-shadow:
        0 5px 15px rgba(15,23,42,0.06);
}

.result-id {
    color: #2563eb;
    font-weight: 800;
}

.badge {
    display: inline-block;

    padding: 5px 12px;

    border-radius: 20px;

    background: #ede9fe;

    color: #6d28d9;

    font-size: 11px;

    font-weight: 700;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #eef4ff,
        #f8f1ff
    );
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #172554;
}

/* Upload box */

[data-testid="stFileUploader"] {
    background: white;
    border-radius: 14px;
    border: 2px dashed #8b5cf6;
    padding: 5px;
}

/* Download button */

.stDownloadButton button {
    width: 100%;

    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );

    color: white;

    border: none;

    border-radius: 12px;

    font-weight: 700;

    padding: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

<div style="display:flex;justify-content:space-between;align-items:center;">

<div>

<div class="hero-title">
🔍 METALVISION AI
</div>

<div class="hero-subtitle">
Automated Metal Component Defect Inspection
</div>

</div>

<div class="online">
🟢 SYSTEM ONLINE
</div>

</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎛️ Inspection Control")

st.sidebar.caption(
    "Computer Vision Inspection System"
)

uploaded_file = st.sidebar.file_uploader(
    "📤 Upload Component Image",
    type=["jpg", "jpeg", "png", "webp"]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### ⚙️ Processing Pipeline"
)

st.sidebar.success("🟢 Image Acquisition")
st.sidebar.success("🟢 Preprocessing")
st.sidebar.success("🟢 ROI Detection")
st.sidebar.success("🟢 Defect Segmentation")
st.sidebar.success("🟢 Feature Extraction")
st.sidebar.success("🟢 Classification")


# =========================================================
# WELCOME SCREEN
# =========================================================

if uploaded_file is None:

    st.markdown("""
    <div style="
        background:white;
        padding:55px;
        border-radius:22px;
        text-align:center;
        box-shadow:0 8px 25px rgba(0,0,0,0.07);
        border:1px solid #e2e8f0;
    ">

    <div style="font-size:60px;">
    🔍
    </div>

    <h2 style="color:#172554;">
    Ready for Inspection
    </h2>

    <p style="
        color:#64748b;
        font-size:16px;
    ">
    Upload a metal component image from the
    inspection panel to begin automated analysis.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">🚀 System Pipeline</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(9)

    steps = [
        ("01", "📷", "ACQUIRE"),
        ("02", "⚙️", "PROCESS"),
        ("03", "🎯", "ROI"),
        ("04", "🔎", "SEGMENT"),
        ("05", "📐", "FEATURES"),
    ]

    for i, (number, icon, label) in enumerate(steps):

        with cols[i * 2]:

            st.markdown(
                f"""
                <div class="pipeline-card">

                <div class="pipeline-step">

                <div class="pipeline-number">
                {number}
                </div>

                <div style="font-size:25px;margin-top:8px;">
                {icon}
                </div>

                <div class="pipeline-label">
                {label}
                </div>

                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        if i < len(steps) - 1:

            with cols[i * 2 + 1]:

                st.markdown(
                    '<div class="pipeline-arrow">→</div>',
                    unsafe_allow_html=True
                )

    st.stop()


# =========================================================
# SAVE IMAGE
# =========================================================

os.makedirs("test_images", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

input_path = os.path.join(
    "test_images",
    uploaded_file.name
)

with open(input_path, "wb") as file:

    file.write(
        uploaded_file.getbuffer()
    )


# =========================================================
# RUN PIPELINE
# =========================================================

with st.spinner(
    "🔄 Running MetalVision AI inspection..."
):

    preprocess_image(input_path)

    features = segment_defects(
        "outputs/enhanced.jpg"
    )

    results = generate_report(
        features
    )


# =========================================================
# METRICS
# =========================================================

total = len(results)

high = sum(
    r["severity"] == "High"
    for r in results
)

medium = sum(
    r["severity"] == "Medium"
    for r in results
)

low = sum(
    r["severity"] == "Low"
    for r in results
)


st.markdown(
    '<div class="section-title">📊 Inspection Overview</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

cards = [
    ("🔎", "CANDIDATES", total, "blue"),
    ("🚨", "HIGH SEVERITY", high, "red"),
    ("⚠️", "MEDIUM", medium, "orange"),
    ("✅", "LOW", low, "green")
]

for col, (icon, title, value, style) in zip(
    [c1, c2, c3, c4],
    cards
):

    with col:

        st.markdown(
            f"""
            <div class="metric-card {style}">

            <div class="metric-title">
            {icon} {title}
            </div>

            <div class="metric-number">
            {value}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# IMAGE WORKSPACE
# =========================================================

st.markdown(
    '<div class="section-title">🖼️ Inspection Workspace</div>',
    unsafe_allow_html=True
)

c1, c2 = st.columns(2)

with c1:

    st.markdown(
        '<div class="image-card"><div class="image-title">📷 Original Image</div>',
        unsafe_allow_html=True
    )

    st.image(
        input_path,
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


with c2:

    st.markdown(
        '<div class="image-card"><div class="image-title">✨ Enhanced Image</div>',
        unsafe_allow_html=True
    )

    st.image(
        "outputs/enhanced.jpg",
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


c3, c4 = st.columns(2)

with c3:

    st.markdown(
        '<div class="image-card"><div class="image-title">🎯 Component ROI</div>',
        unsafe_allow_html=True
    )

    if os.path.exists(
        "outputs/component_roi.jpg"
    ):

        st.image(
            "outputs/component_roi.jpg",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)


with c4:

    st.markdown(
        '<div class="image-card"><div class="image-title">🔎 Defect Candidates</div>',
        unsafe_allow_html=True
    )

    st.image(
        "outputs/defect_detection.jpg",
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# RESULTS
# =========================================================

st.markdown(
    '<div class="section-title">📋 Inspection Results</div>',
    unsafe_allow_html=True
)

if results:

    for result in results:

        st.markdown(
            f"""
            <div class="result-card">

            <span class="result-id">
            Candidate {result['candidate_id']:02d}
            </span>

            &nbsp;&nbsp;

            <span class="badge">
            {result['classification']}
            </span>

            &nbsp;

            <span class="badge">
            {result['severity']} Severity
            </span>

            <br><br>

            <span style="color:#64748b;">
            Area: <b>{result['area']}</b>
            &nbsp;&nbsp;
            Perimeter: <b>{result['perimeter']}</b>
            &nbsp;&nbsp;
            Aspect Ratio: <b>{result['aspect_ratio']}</b>
            &nbsp;&nbsp;
            Circularity: <b>{result['circularity']}</b>
            </span>

            </div>
            """,
            unsafe_allow_html=True
        )

else:

    st.success(
        "✅ No potential defect candidates detected."
    )


# =========================================================
# REPORT
# =========================================================

report = {
    "image": uploaded_file.name,
    "total_candidates": total,
    "high_severity": high,
    "medium_severity": medium,
    "low_severity": low,
    "results": results
}

report_json = json.dumps(
    report,
    indent=4
)

st.markdown(
    '<div class="section-title">📄 Inspection Report</div>',
    unsafe_allow_html=True
)

st.download_button(
    "📥 Download Inspection Report",
    report_json,
    "metalvision_inspection_report.json",
    "application/json"
)