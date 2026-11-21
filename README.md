# Smart Document Scanner

**Course:** Artificial Intelligence (25MCAC104), MCA Semester I, JAIN University Kochi
**Topic:** Computer Vision mini project

The Smart Document Scanner turns an ordinary phone photo of a page into a clean,
flat, scanned-looking black-and-white image. It uses only classical computer vision
(no deep learning). It finds the page's outline with edge detection and contour
analysis, locates its four corners with a fallback chain for hard photos, removes
the perspective tilt with a homography, and cleans the result with adaptive
thresholding. An evaluation script measures how accurately the page is found on
our own test photos. It compares each detected outline with hand-clicked ground
truth using Intersection over Union (IoU), reports the success rate per photo
condition, and explains every failure. Optional OCR (Tesseract) extracts editable
text and shows that scanning lowers the OCR error rate.

---

## 1. Setup

Requires Python 3.9 or newer (3.10+ recommended).

```bash
cd smart-document-scanner
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Optional: Tesseract OCR engine

`pytesseract` is only a wrapper; the Tesseract program must be installed separately.
Everything else works without it.

| OS | Command |
|---|---|
| macOS | `brew install tesseract` |
| Ubuntu / Debian | `sudo apt install tesseract-ocr` |
| Windows | Install from the UB Mannheim build: https://github.com/UB-Mannheim/tesseract/wiki, then add `C:\Program Files\Tesseract-OCR` to `PATH` |

Check with `tesseract --version`.

### Check that everything works

```bash
python -m pytest -v          # 24 tests on synthetic images
python make_samples.py       # 12 synthetic demo photos in data/samples/
python scan.py data/samples/sample_02.jpg --save-stages
```

---

## 2. Photo collection guide (20+ photos)

The evaluation is only as good as the test set. **Deliberately include hard
cases.** A test set where everything succeeds gives an empty failure analysis
and a weak report. Aim for about 25 photos spread over these conditions, and
vary one condition at a time where you can:

| Condition | Values (use exactly these words in `conditions.csv`) | How to shoot |
|---|---|---|
| **angle** | `straight`, `tilted`, `steep` | straight = phone directly above; tilted = ~20–30°; steep = ~45°+, far edge clearly shorter |
| **lighting** | `good`, `dim`, `shadow`, `glare` | dim = lights off / evening; shadow = your hand or phone casting a shadow across the page; glare = glossy paper under a lamp |
| **background** | `plain`, `cluttered`, `low-contrast` | plain = dark desk; cluttered = pens, books, phone around (and touching) the page; low-contrast = **white page on a white table or bedsheet** |
| **page** | `flat`, `curled`, `folded` | curled = page from a thick book or rolled paper; folded = a letter folded in thirds and opened |

Suggested mix: about 8 "easy" photos (straight/tilted, good light, plain, flat)
and 2–3 photos for each hard value above. Some photos can combine hard
conditions (e.g. steep + shadow).

Practical rules:
- **All four corners of the page must be visible** in the photo (otherwise there is no ground truth to click).
- Use printed text pages (a handout, a book page), so the scans and OCR are meaningful.
- Save as **JPEG**. iPhone HEIC files cannot be read by OpenCV: set *Settings → Camera → Formats → Most Compatible*, or export as JPEG.
- Give files simple names (e.g. `IMG_01.jpg`, no spaces), and copy them into `data/raw/`.
- Optional, for the OCR evaluation: for 3–5 photos, type the page's exact text into `data/ocr_truth/<photo name without extension>.txt` (e.g. `data/ocr_truth/IMG_01.txt`).

---

## 3. Workflow

```bash
# 1. Collect photos  -> copy them into data/raw/

# 2. Click the 4 true page corners of every photo (resumes where you stopped)
python annotate.py
#    left-click = corner | u = undo | r = reset | Enter = save | s = skip | q = quit

# 3. Create the conditions table, then fill it in by hand (Excel/Sheets/text editor)
python annotate.py --conditions-template
#    -> data/conditions.csv: angle, lighting, background, page for each photo

# 4. Evaluate: success rate, per-condition table, figures, failure analysis
python evaluate.py
python evaluate.py --ocr            # also OCR CER evaluation (needs Tesseract + data/ocr_truth/)

# 5. Read results/failure_analysis.md, check each auto-generated cause against
#    results/figures/fig3_failure_gallery.png and edit the text by hand.

# 6. Demo
streamlit run app.py
```

To try the whole workflow on the synthetic samples first:

```bash
python make_samples.py
python evaluate.py --images data/samples --gt data/samples/ground_truth.csv \
    --conditions data/samples/conditions.csv --out results/samples
```

### Command reference

| Command | What it does |
|---|---|
| `python scan.py photo.jpg` | Scan one photo → `output/photo_scan.png` (B/W) and `output/photo_color.png` |
| `python scan.py data/raw/ --batch` | Scan a whole folder |
| `--save-stages` | Also save the 6 intermediate stages to `output/stages/` |
| `--ocr` | Run OCR and save `output/photo_scan.txt` |
| `--no-threshold` | Save only the flattened colour scan |
| `--canny-low 50 --canny-high 150 --block-size 31 --c 12` | Override parameters |
| `python evaluate.py --iou 0.85` | Change the success threshold (default 0.90) |
| `python evaluate.py --sample IMG_07.jpg` | Choose the photo used for the pipeline-stages figure |
| `python annotate.py --redo IMG_07.jpg` | Re-click one photo |
| `python ocr.py output/photo_scan.png` | OCR an existing image |

### What `evaluate.py` produces (everything the report needs)

| File | Use in report |
|---|---|
| `results/results.csv` | Full per-photo table: method, IoU, success, conditions, time |
| `results/summary.md` | Overall success rate, mean IoU, tables by detection method and by condition (Markdown, paste-ready) |
| `results/summary_by_condition.csv`, `summary_by_method.csv` | Same tables as CSV |
| `results/failure_analysis.md` | Every failure with conditions and an auto-generated likely cause. **Edit by hand.** Re-running `evaluate.py` does not overwrite your edits; it writes `failure_analysis_auto.md` instead (use `--overwrite-analysis` to replace) |
| `figures/fig1_pipeline_stages.png` | The 6 stages for one photo |
| `figures/fig2_before_after.png` | 6 successful scans, before/after |
| `figures/fig3_failure_gallery.png` | All failures, detected quad in red, ground truth in green |
| `figures/fig4_success_by_condition.png` | Success rate by condition |
| `figures/fig5_iou_distribution.png` | IoU of every photo vs. the threshold |
| `figures/fig6_ocr_cer.png`, `ocr_cer.csv`, `ocr_summary.txt` | OCR character error rate, raw photo vs scan (with `--ocr`) |

---

## 4. How the pipeline works

Code: `scanner/` package; `scanner/pipeline.py: scan_document()` runs these steps in order.

**Step 1 – Resize for detection** (`preprocess.resize_for_detection`).
Phone photos are 3000–4000 px tall. We make a copy that is 500 px tall for
finding the page. This makes detection about 40× faster, and fine texture such as
paper grain or wood grain mostly disappears at this size. We remember the scale
`ratio = original height / 500`, so the corners found on the small copy can be
multiplied back to full resolution. The final scan is cut from the original photo
and keeps full quality.

**Step 2 – Grayscale and Gaussian blur** (`to_grayscale`, `blur`).
Colour is not needed to find edges, so we keep only brightness. A 5×5 Gaussian
blur averages each pixel with its neighbours (weighted by a bell curve). This
removes camera noise and small texture that would otherwise create thousands of
tiny false edges.

**Step 3 – Canny edge detection and closing** (`detect_edges`, `close_edges`).
Canny finds places where brightness changes sharply. It uses two thresholds:
gradients above 200 are certain edges, below 75 are ignored, and values in
between are kept only if they connect to a certain edge. The border between a
white page and a darker desk is a strong, long edge, so it survives. A small
morphological *closing* (dilate, then erode, with a 5×5 square) and one 3×3
dilation then join short gaps, so the page border becomes one closed line.

**Step 4 – Contours and the 4-corner test** (`detect.find_quadrilateral`).
`findContours` traces the outline of every connected shape in the edge image. We
sort the outlines by enclosed area and check the 5 largest. Each one is
simplified with `approxPolyDP` (the Douglas–Peucker algorithm), which removes
points until the shape stays within a tolerance of 2% of its perimeter. A page
seen at an angle simplifies to exactly 4 points. We accept the first outline that:
- has exactly 4 corners;
- is convex (a flat sheet in perspective always is);
- covers at least 20% of the photo (anything smaller is probably a text block or a sticky note) and at most 98% (that would be the photo border itself);
- is at most 20% longer than its 4-point simplification (rejects ragged blobs where the page edge merged with background texture).

**Step 5 – Fallback chain** (`detect.detect_page`).
If Canny finds no valid outline, we try simpler ideas in turn and record which one worked:
1. **`otsu`:** Otsu's method picks the brightness threshold that best splits
   the image into two groups ("page" and "background"). A 9×9 closing fills the
   text holes, and Step 4 runs on this mask. It works on whole regions, so it
   helps when the border edge is soft or broken.
2. **`minarearect`:** the smallest rotated rectangle around the largest shape.
   This always gives a box, but a rectangle cannot follow perspective, so
   accuracy is lower (used for curled or partly hidden pages).
3. **`failed`:** nothing usable was found. The program reports "page not detected" and does not crash.

**Step 6 – Order the corners** (`transform.order_points`).
Contours list corners in arbitrary order. We sort them clockwise by their angle
around the page centre and start from the corner nearest the image's top-left.
The result is always top-left, top-right, bottom-right, bottom-left.

**Step 7 – Perspective transform** (`transform.four_point_transform`).
The output width is the longer of the page's top and bottom edges, and the height
is the longer of its left and right edges. `getPerspectiveTransform` computes the
3×3 homography matrix that maps the 4 page corners onto the 4 corners of a
rectangle of that size. `warpPerspective` then moves every pixel of the **original
full-resolution photo** accordingly. The result looks as if the page had been
photographed from directly above. This flattened image is saved as the *colour scan*.

**Step 8 – Adaptive threshold** (`enhance.adaptive_threshold`).
A single global threshold fails when one part of the page is in shadow. Adaptive
thresholding computes a separate threshold for every pixel: the Gaussian-weighted
average brightness of its 25×25 neighbourhood (`blockSize`) minus 10 (`C`).
Darker than that means ink (black); otherwise it is paper (white). `blockSize`
must be bigger than the thickness of a letter stroke. A larger `C` gives a cleaner
background but thinner text.

### How accuracy is measured

For each photo, we draw the detected quadrilateral and the hand-clicked one as
filled masks. IoU = overlapping pixels / pixels covered by either shape. An IoU of
1.0 is a perfect match. A photo counts as a **success if IoU ≥ 0.90**. The success
rate is reported overall, per condition and per detection method.
For OCR, the **Character Error Rate** is the edit (Levenshtein) distance between
the OCR text and the true text, divided by the length of the true text. We compute
it with our own dynamic-programming code in `ocr.py`.

### Parameter summary

| Parameter | Default | Where | Why this value |
|---|---|---|---|
| Detection height | 500 px | preprocess | Fast, and removes fine texture |
| Gaussian kernel | 5×5 | preprocess | Removes pixel noise without rounding page corners |
| Canny low / high | 75 / 200 | preprocess | Page border is a strong edge; weak texture is ignored; ~1:2.7 ratio |
| Edge closing | 5×5 + 3×3 dilate | preprocess | Bridges gaps up to ~5 px in the border |
| Top contours checked | 5 | detect | The page is among the largest shapes |
| approxPolyDP epsilon | 2% of perimeter | detect | Straightens wobbly edges but keeps real corners |
| Min / max page area | 20% / 98% | detect | Rejects small inner shapes and the image border |
| Max perimeter ratio | 1.2 | detect | Rejects ragged page+texture blobs |
| Otsu closing | 9×9 | preprocess | Fills text lines so the page is one solid region |
| blockSize / C | 25 / 10 | enhance | Larger than stroke width; removes paper texture |
| IoU success threshold | 0.90 | evaluate | Corners within a few % of the page size |

---

## 5. Project structure

```
smart-document-scanner/
├── scanner/              the pipeline, one module per stage
│   ├── preprocess.py     resize, grayscale, blur, Canny, closing, Otsu
│   ├── detect.py         contour search + fallback chain
│   ├── transform.py      corner ordering + perspective warp
│   ├── enhance.py        adaptive thresholding
│   ├── pipeline.py       scan_document() -> ScanResult
│   ├── metrics.py        polygon IoU (mask-based)
│   └── synthetic.py      synthetic test-photo generator
├── scan.py               command-line scanner
├── annotate.py           ground-truth click tool + conditions template
├── evaluate.py           evaluation, tables, figures, failure analysis
├── ocr.py                optional Tesseract OCR + CER evaluation
├── app.py                Streamlit demo
├── make_samples.py       creates data/samples/
├── tests/test_pipeline.py
├── data/raw/             YOUR photos go here
├── data/ocr_truth/       optional typed text for OCR evaluation
├── data/ground_truth.csv written by annotate.py
├── data/conditions.csv   filled in by hand
├── data/samples/         synthetic demo photos (+ their own CSVs)
├── results/              evaluation outputs
└── output/               scans from scan.py
```

## 6. Troubleshooting

- **`annotate.py` window does not appear (macOS):** run it from a normal Terminal, not a remote session. It needs `opencv-python`, not `opencv-python-headless`.
- **"could not read image":** probably a HEIC file. Convert it to JPEG.
- **Rotated corners in `annotate.py`:** both tools load photos with `cv2.imread`, which applies the phone's EXIF rotation, so clicks and detections share the same coordinates. If you rotated photos in another program, re-run `annotate.py --redo <file>`.

See `REFERENCES.md` for the sources adapted in this project.
