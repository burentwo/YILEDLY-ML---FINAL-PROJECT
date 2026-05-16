<p align="center">
  <img src="yieldlylogo111.png" alt="Yieldly Logo" width="320"/>
</p>


A mobile-first crop disease detection tool built for Filipino eggplant farmers. Yieldly uses a fine-tuned EfficientNet-B2 model to classify eggplant images into five disease categories and delivers results in plain language — in either English or Filipino — within seconds of upload.

**It helps users:**
- Upload or camera-capture photos of eggplant leaves, fruits, or shoots for instant disease diagnosis
- Receive a two-phase diagnosis (health check first, then disease identification if needed)
- View recommended treatment actions and prevention tips per detected disease
- Export a per-scan diagnosis report or a full session summary as a downloadable PDF
- Switch the entire interface between English and Filipino at any time
- Review their full diagnosis history for the current session and remove individual entries

---

## Tech Stack

Yieldly runs entirely as a single-file Streamlit application. The ML model is loaded from a local `.pth` file at startup; if the file is absent, the app enters Demo Mode with mock inference so the interface remains testable without the weights.

- **Frontend / UI:** Streamlit (PWA-compatible, mobile-responsive CSS overrides included)
- **ML Framework:** PyTorch + `timm` (`efficientnet_b2`, transfer learning, 7 output classes)
- **Inference strategy:** Test-Time Augmentation — original image + horizontal flip + vertical flip, logits averaged before softmax
- **Image processing:** Pillow (`PIL`), NumPy
- **PDF generation:** `fpdf2` (per-scan diagnosis report + session summary report)
- **Fonts:** Poppins (headings), Inter (body) via Google Fonts
- **Language support:** English and Filipino (full string translation dictionary in `app.py`)

---

## Current App Flow

On load, the sidebar renders a language selector (English / Filipino), a collapsible model info panel listing all detectable classes and the two-phase diagnosis logic, and a scrollable session history showing the five most recent scans.

The main panel displays the Yieldly logo, a subtitle in the active language, and a mode banner — either a live-inference confirmation or a Demo Mode notice if `yieldy_model(matrix2).pth` is missing.

The upload panel presents two tabs: **Upload** (multi-file, JPG/JPEG/PNG) and **Quick Capture** (live camera input with a 4:3 crop). When multiple images are loaded, left/right navigation buttons let the user step through results one at a time.

For each image, the app runs the full inference pipeline and displays the diagnosis card. After all results are rendered, a disclaimer banner reminds the user to verify findings with a local agronomist. A session-level PDF summary export button is available at the bottom of the history panel.

**Main working features right now:**
- Multi-image upload with per-image navigation (Previous / Next controls)
- Live camera capture tab with automatic 4:3 aspect-ratio crop
- Two-phase diagnosis gate: Phase 1 health check (≥ 80% healthy probability required to pass), Phase 2 disease detection if Phase 1 fails
- Invalid image filter rejecting uploads that fall below the confidence threshold or exceed entropy threshold
- Confidence breakdown bar chart across all seven classes for every scan
- Severity banner (High / Medium / None) with color-coded styling per result
- Inline feedback buttons ("Yes, accurate" / "No, inaccurate") per diagnosis
- Per-scan PDF report export (diagnosis, cause, symptom, actions, prevention, confidence breakdown)
- Session summary PDF export (total scans, healthy count, disease count, per-scan table)
- Full session history panel with timestamp, severity, diagnosis name, and confidence; individual entry removal and bulk clear
- Demo Mode with deterministic mock inference when model weights are absent

---

## Important Notes

| Parameter | Value | Effect |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.55` | Images where the top predicted class scores below this are rejected as invalid/unrecognized |
| `HEALTHY_GATE_THRESHOLD` | `0.80` | Phase 1 passes (plant marked healthy) only when the combined probability of `Eggplant Healthy Fruit` + `Eggplant Healthy Leaf` is ≥ 80% |
| `ENTROPY_THRESHOLD` | `1.80` | Images whose softmax distribution entropy exceeds this value are rejected regardless of top-class confidence |
| Model filename | `yieldy_model(matrix2).pth` | Must be placed in the **same directory** as `app.py`; any other name or location causes a fallback to Demo Mode |
| TTA variants | 3 (original, H-flip, V-flip) | Logits from all three passes are summed and averaged before softmax — do not change image size from `260×260` without retraining |
| `HEALTHY_CLASSES` | `{"Eggplant Healthy Fruit", "Eggplant Healthy Leaf"}` | Both classes contribute to the Phase 1 healthy probability pool; the combined score is what is compared against the gate threshold |
| Session history cap (sidebar) | 5 entries displayed | Only the five most recent records are shown in the sidebar; the full list is visible in the history panel below the main content |
| Demo Mode inference | Deterministic via image mean seed | Mock scores are seeded from `int(img_array.mean() * 100)` so the same image always produces the same demo output |

---

## Folder Guide

```
yieldly/
├── app.py                        # Entire application — UI, inference, PDF generation, translation strings
├── yieldlylogo111.png            # Logo asset; loaded at runtime via base64 embed into the header
├── yieldy_model(matrix2).pth     # EfficientNet-B2 weights (NOT included in repo — obtain separately)
└── requirements.txt              # Python dependencies (see Quick Setup)
```

> `All files must be in the same folder when you run `streamlit run app.py`.

---

## Files Needed

When running this project in a local environment, include the following files:

- `app.py` — complete application source
- `yieldlylogo111.png` — logo asset required at runtime
- `yieldy_model(matrix2).pth` — trained EfficientNet-B2 weights (share via drive or LFS; do not commit to the repo)
- `requirements.txt` — pinned Python dependencies

---

## Quick Setup for Team Members

```bash
# 1. Clone the repository
git clone https://github.com/your-org/yieldly.git
cd yieldly

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the model weights in the project root
#    File must be named exactly: yieldy_model(matrix2).pth
#    (Without it, the app runs in Demo Mode — the UI still works for testing)

# 5. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501` by default. On first load, `@st.cache_resource` caches the model in memory — subsequent uploads do not reload weights.

**Minimum Python version:** 3.10+ (uses `tuple[...]` type hints and `bytes | None` union syntax)

**Key dependencies:**
- `streamlit`
- `torch`
- `torchvision`
- `timm`
- `Pillow`
- `numpy`
- `fpdf2`

---

## Project Contributors

- Brent Dela Cruz
- Ethan Tyler Razalan
- Gabriel John Uy
