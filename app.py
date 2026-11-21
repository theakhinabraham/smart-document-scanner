"""Streamlit demo for the Smart Document Scanner.

Run with:
    streamlit run app.py

Upload a photo (or pick one from data/raw or data/samples), move the sliders
and watch the detection and the scan update live.
"""

import tempfile
from pathlib import Path

import cv2
import streamlit as st

from scanner import draw_outline, load_image, scan_document
from scanner.pipeline import STAGE_NAMES, list_images

st.set_page_config(page_title="Smart Document Scanner", layout="wide")

METHOD_TEXT = {
    "canny": "Canny edges (primary method)",
    "otsu": "Otsu threshold (fallback 1)",
    "minarearect": "minAreaRect (fallback 2 - rough box, no perspective)",
    "failed": "page not detected",
}


def rgb(image):
    """OpenCV images are BGR; Streamlit expects RGB."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False)
def read_upload(data, name):
    """Decode an uploaded file via a temporary file so load_image (and EXIF rotation) is reused."""
    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        return load_image(tmp.name)


# --- Sidebar: input + parameters ------------------------------------------------
st.sidebar.header("Input")
folders = {name: list_images(f"data/{name}") for name in ("raw", "samples")}
sources = ["Upload a photo"] + [f"Pick from data/{n}" for n, files in folders.items() if files]
source = st.sidebar.radio("Source", sources)

image, name = None, None
if source == "Upload a photo":
    uploaded = st.sidebar.file_uploader("Photo", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"])
    if uploaded is not None:
        image, name = read_upload(uploaded.getvalue(), uploaded.name), uploaded.name
else:
    folder = source.split("/")[-1]
    chosen = st.sidebar.selectbox("Photo", folders[folder], format_func=lambda p: p.name)
    image, name = load_image(chosen), chosen.name

st.sidebar.header("Edge detection (Canny)")
canny_low = st.sidebar.slider("Lower threshold", 0, 255, 75,
                              help="Weak edges between the two thresholds are kept only if they touch a strong edge.")
canny_high = st.sidebar.slider("Upper threshold", 0, 255, 200,
                               help="Gradients above this are definitely edges.")
st.sidebar.header("Adaptive threshold")
block_size = st.sidebar.slider("blockSize (odd)", 3, 101, 25, step=2,
                               help="Neighbourhood size used to compute each pixel's local threshold.")
c_value = st.sidebar.slider("C", 0, 40, 10,
                            help="Subtracted from the local mean: larger = cleaner background, thinner text.")

# --- Main page ------------------------------------------------------------------
st.title("Smart Document Scanner")
st.caption("Classical computer vision: edges -> contours -> perspective transform -> adaptive threshold")

if image is None:
    st.info("Upload a photo or pick one in the sidebar. Run `python make_samples.py` for demo images.")
    st.stop()

result = scan_document(image, canny_low=canny_low, canny_high=max(canny_high, canny_low),
                       block_size=block_size, c=c_value, keep_stages=True)

if result.success:
    st.success(f"Detection method: **{METHOD_TEXT[result.method]}**  ·  {result.time_ms:.0f} ms")
else:
    st.error(f"{result.message}. Try lowering the Canny thresholds, or retake the photo "
             "on a darker, plain background.")

col1, col2, col3 = st.columns(3)
col1.image(rgb(draw_outline(image, result.corners)), caption="1. Original + detected outline",
           width="stretch")
if result.success:
    col2.image(rgb(result.warped), caption="2. Flattened (perspective transform)", width="stretch")
    col3.image(result.scanned, caption="3. Adaptive threshold (final scan)", width="stretch")

with st.expander("All intermediate stages"):
    stage_cols = st.columns(3)
    for i, key in enumerate(n for n in STAGE_NAMES if n in result.stages):
        stage_cols[i % 3].image(rgb(result.stages[key]), caption=key, width="stretch")

if result.success:
    st.subheader("Export")
    left, right = st.columns(2)
    ok, png = cv2.imencode(".png", result.scanned)
    left.download_button("Download black-and-white scan", png.tobytes(),
                         file_name=f"{Path(name).stem}_scan.png", mime="image/png")
    ok, png = cv2.imencode(".png", result.warped)
    right.download_button("Download colour scan", png.tobytes(),
                          file_name=f"{Path(name).stem}_color.png", mime="image/png")

    st.subheader("OCR (optional)")
    if st.button("Extract text with Tesseract"):
        from ocr import INSTALL_HINT, extract_text, tesseract_available
        if not tesseract_available():
            st.warning(INSTALL_HINT)
        else:
            with st.spinner("Running OCR..."):
                text = extract_text(result.scanned)
            st.text_area("Recognised text", text, height=250)
