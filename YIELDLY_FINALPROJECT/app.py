# IMPORTS

import io
import os
import time
import hashlib
import datetime

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# THRESHOLDS

CONFIDENCE_THRESHOLD   = 0.55
HEALTHY_GATE_THRESHOLD = 0.80
ENTROPY_THRESHOLD      = 1.80

HEALTHY_CLASSES = {"Eggplant Healthy Fruit", "Eggplant Healthy Leaf"}

# TRANSLATIONS  (English / Filipino)

LANG_OPTIONS = ["English", "Filipino"]

T = {
    "app_title": {
        "English":  "🍆 YIELDLY",
        "Filipino": "🍆 YIELDLY",
    },
    "app_subtitle": {
        "English":  "A crop disease detection tool for Filipino Eggplant Farmers",
        "Filipino": "Isang kagamitan sa pagtuklas ng sakit ng tanim para sa mga magsasakang Pilipino",
    },
    "upload_header": {
        "English":  "Upload Photo(s)",
        "Filipino": "Mag-upload ng Larawan",
    },
    "upload_instruction": {
        "English":  "Upload one or more eggplant images (leaf, fruit, or shoot) to begin.",
        "Filipino": "Mag-upload ng isa o higit pang larawan ng talong (dahon, bunga, o sanga).",
    },
    "supported_formats": {
        "English":  "Supports JPG, JPEG, PNG · Best results with clear, close-up photos.",
        "Filipino": "Tinatanggap ang JPG, JPEG, PNG · Pinakamabuting resulta sa malinaw na larawan.",
    },
    "model_loaded": {
        "English":  "EfficientNet-B2 model loaded — running live inference.",
        "Filipino": "Na-load na ang modelo ng EfficientNet-B2 — nagtatakbo ang live inference.",
    },
    "demo_mode": {
        "English":  "**Demo Mode** — Upload any eggplant photo for a simulated diagnosis. Place `yieldy_model.pth` next to `app.py` for real inference.",
        "Filipino": "**Demo Mode** — Mag-upload ng larawan ng talong para sa simulate na diagnosis. Ilagay ang `yieldy_model.pth` katabi ng `app.py` para sa tunay na inference.",
    },
    "analyzing": {
        "English":  "Running analysis...",
        "Filipino": "Sinusuri...",
    },
    "invalid_image_title": {
        "English":  "Invalid or Unrecognized Image",
        "Filipino": "Hindi Makilala ang Larawan",
    },
    "invalid_image_body": {
        "English":  (
            "This image does not appear to be an eggplant leaf or fruit. "
            "Please upload a clear, close-up photo of an eggplant plant part "
            "(leaf, fruit, or shoot) for an accurate diagnosis."
        ),
        "Filipino": (
            "Ang larawang ito ay hindi mukhang dahon o bunga ng talong. "
            "Mangyaring mag-upload ng malinaw, malapit na larawan ng bahagi ng talong "
            "(dahon, bunga, o sanga) para sa tumpak na diagnosis."
        ),
    },
    "phase1_passed": {
        "English":  "Phase 1 Passed - Plant Appears Healthy",
        "Filipino": "Naipasa ang Phase 1 - Mukhang Malusog ang Halaman",
    },
    "phase2_label": {
        "English":  "Phase 2 - Disease Identified",
        "Filipino": "Phase 2 - Natukoy ang Sakit",
    },
    "phase1_fail": {
        "English":  "Your eggplant doesn't appear to be healthy. Proceeding to Phase 2 - Disease Detection.",
        "Filipino": "Ang inyong talong ay mukhang may sakit. Nagpapatuloy sa Phase 2 - Pagtuklas ng Sakit.",
    },
    "confidence_label": {
        "English":  "Confidence",
        "Filipino": "Katiyakan",
    },
    "observation": {
        "English":  "Observation",
        "Filipino": "Obserbasyon",
    },
    "primary_symptom": {
        "English":  "Primary Symptom",
        "Filipino": "Pangunahing Sintomas",
    },
    "severity_label": {
        "English":  "Severity",
        "Filipino": "Kalubhaan",
    },
    "healthy_prob": {
        "English":  "Healthiness Probability",
        "Filipino": "Probabilidad  na Malusog",
    },
    "banner_high": {
        "English":  "HIGH SEVERITY - Immediate action required. Do not delay treatment.",
        "Filipino": "MATAAS NA KALUBHAAN - Kailangan ng agarang aksyon. Huwag patagalin ang lunas.",
    },
    "banner_medium": {
        "English":  "MEDIUM SEVERITY - Monitor closely and begin treatment soon.",
        "Filipino": "KATAMTAMANG KALUBHAAN - Bantayan nang mabuti at simulan ang lunas.",
    },
    "banner_none": {
        "English":  "NO DISEASE DETECTED - Your plant appears healthy. Keep up good farming practices.",
        "Filipino": "WALANG SAKIT NA NATUKOY - Mukhang malusog ang iyong halaman. Ituloy ang mabuting pagsasaka.",
    },
    "recommended_actions": {
        "English":  "Recommended Actions",
        "Filipino": "Mga Inirerekomendang Aksyon",
    },
    "prevention_tips": {
        "English":  "Prevention Tips",
        "Filipino": "Mga Tip sa Pag-iwas",
    },
    "confidence_breakdown": {
        "English":  "Confidence Breakdown - All Classes",
        "Filipino": "Breakdown ng Katiyakan - Lahat ng Klase",
    },
    "feedback_prompt": {
        "English":  "Is this diagnosis accurate?",
        "Filipino": "Tama ba ang diagnosis na ito?",
    },
    "feedback_yes": {
        "English":  "Yes, accurate",
        "Filipino": "Oo, tama",
    },
    "feedback_no": {
        "English":  "No, inaccurate",
        "Filipino": "Hindi, mali",
    },
    "feedback_thanks": {
        "English":  "Thank you for your feedback!",
        "Filipino": "Salamat sa inyong feedback!",
    },
    "export_pdf": {
        "English":  "Export Diagnosis Report (PDF)",
        "Filipino": "I-export ang Ulat ng Diagnosis (PDF)",
    },
    "pdf_filename": {
        "English":  "yieldly_diagnosis_report.pdf",
        "Filipino": "yieldly_diagnosis_report.pdf",
    },
    "export_pdf_summary": {
        "English":  "Export Session Summary Report (PDF)",
        "Filipino": "I-export ang Buod ng Session (PDF)",
    },
    "pdf_summary_filename": {
        "English":  "yieldly_summary_report.pdf",
        "Filipino": "yieldly_summary_report.pdf",
    },
    "pdf_no_history": {
        "English":  "No history to export. Run at least one diagnosis first.",
        "Filipino": "Walang kasaysayan para i-export. Magsagawa muna ng isang diagnosis.",
    },
    "pdf_summary_title": {
        "English":  "Yieldly Disease Diagnosis Summary Report",
        "Filipino": "Yieldly - Buod ng Ulat ng Diagnosis ng Sakit",
    },
    "pdf_total_scans": {
        "English":  "Total Scans",
        "Filipino": "Kabuuang Scan",
    },
    "pdf_healthy_count": {
        "English":  "Healthy",
        "Filipino": "Malusog",
    },
    "pdf_disease_count": {
        "English":  "Disease Detected",
        "Filipino": "May Sakit",
    },
    "pdf_scan_label": {
        "English":  "Scan",
        "Filipino": "Scan",
    },
    "history_title": {
        "English":  "Session History",
        "Filipino": "Kasaysayan ng Session",
    },
    "history_empty": {
        "English":  "No diagnoses yet this session.",
        "Filipino": "Wala pang diagnosis ngayong session.",
    },
    "history_clear": {
        "English":  "Clear All History",
        "Filipino": "Burahin ang Lahat",
    },
    "history_remove": {
        "English":  "Remove",
        "Filipino": "Alisin",
    },
    "history_total": {
        "English":  "Total diagnoses this session",
        "Filipino": "Kabuuang diagnosis ngayong session",
    },
    "nav_prev": {
        "English":  "Previous",
        "Filipino": "Bumalik",
    },
    "nav_next": {
        "English":  "Next",
        "Filipino": "Sunod",
    },
    "nav_image_of": {
        "English":  "Image {current} of {total}",
        "Filipino": "Larawan {current} ng {total}",
    },
    "disclaimer": {
        "English":  (
            "NOTE: This tool provides a preliminary diagnosis based on visual symptoms "
            "and does not guarantee full accuracy. Always confirm results with a local "
            "agronomist or extension officer."
        ),
        "Filipino": (
            "PAUNAWA: Ang kagamitang ito ay nagbibigay ng paunang diagnosis batay sa "
            "visual na sintomas at hindi ginagarantiyahan ang ganap na katumpakan. "
            "Palaging kumpirmahin ang mga resulta sa isang lokal na agronomist."
        ),
    },
    "sidebar_classes": {
        "English":  "Detectable Classes:",
        "Filipino": "Mga Matatukoy na Klase:",
    },
    "sidebar_logic": {
        "English":  "Diagnosis Logic: Two-Phase System",
        "Filipino": "Lohika ng Diagnosis: Two-Phase System",
    },
    "sidebar_model": {
        "English":  "Model: EfficientNet-B2 (Transfer Learning)",
        "Filipino": "Modelo: EfficientNet-B2 (Transfer Learning)",
    },
    "sidebar_language": {
        "English":  "Language / Wika",
        "Filipino": "Language / Wika",
    },
    "uploaded_image": {
        "English":  "Uploaded image",
        "Filipino": "Na-upload na larawan",
    },
    "tab_upload": {
        "English":  "Upload",
        "Filipino": "Mag-upload",
    },
    "tab_quick_capture": {
        "English":  "Quick Capture",
        "Filipino": "Kumuha ng Litrato",
    },
    "take_a_photo": {
        "English":  "Take a Photo",
        "Filipino": "Kumuha ng Larawan",
    },
    "camera_instruction": {
        "English":  "Point your camera at the eggplant leaf, fruit, or shoot, then tap the capture button.",
        "Filipino": "Itutok ang iyong camera sa dahon, bunga, o sanga ng talong, pagkatapos ay pindutin ang capture button.",
    },
    "camera_photo_ready": {
        "English":  "Photo captured — ready for analysis.",
        "Filipino": "Nakuha ang larawan — handa na para sa pagsusuri.",
    },
    "tta_note": {
        "English":  "TTA applied (original + H-flip + V-flip)",
        "Filipino": "Ginamitan ng TTA (original + H-flip + V-flip)",
    },
}


def tr(key: str, lang: str) -> str:
    """Translate a key to the active language, falling back to English."""
    return T.get(key, {}).get(lang, T.get(key, {}).get("English", key))



# PAGE CONFIG

st.set_page_config(
    page_title="Yieldly - Eggplant Disease Detector",
    page_icon="🍆",
    layout="centered",
)


# GLOBAL CSS

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    h1, h2, h3, h4, h5, h6     { font-family: 'Poppins', sans-serif !important; }

    /* Result cards */
    .yieldly-card {
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }

    /* Severity banner */
    .banner-high   { background:#e74c3c22; border-left:5px solid #e74c3c;
                     padding:12px 16px; border-radius:8px; margin-bottom:14px; }
    .banner-medium { background:#f39c1222; border-left:5px solid #f39c12;
                     padding:12px 16px; border-radius:8px; margin-bottom:14px; }
    .banner-none   { background:#27ae6022; border-left:5px solid #27ae60;
                     padding:12px 16px; border-radius:8px; margin-bottom:14px; }

    /* Confidence bar */
    .conf-bar-wrap { background:#eee; border-radius:8px; height:9px; overflow:hidden; }
    .conf-bar-fill { height:100%; border-radius:8px; transition:width .5s; }

    /* History item */
    .hist-item { font-family:'Inter',sans-serif; font-size:0.80rem;
                 padding:8px 0; border-bottom:1px solid #e0e0e0; }

    /* Navigation bar */
    .nav-bar { display:flex; align-items:center; justify-content:center;
               gap:16px; padding:10px 0; margin-bottom:12px; }

    /* Phase cards — stack cleanly, no overlap */
    .phase-card {
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-sizing: border-box;
        width: 100%;
    }

    /* ── Mobile responsive ──────────────────────────────────────────────── */
    @media (max-width: 768px) {
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        h1 { font-size: 1.8rem !important; }
        .stButton > button { width: 100%; }
        .phase-card { padding: 12px 14px; }
        .conf-bar-wrap { height: 11px; }
        [data-testid="metric-container"] { min-width: 45%; }
    }
    @media (max-width: 480px) {
        h1 { font-size: 1.5rem !important; }
        .phase-card { padding: 10px 12px; }
    }

    [data-testid="stCameraInput"] {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
    }

    /* Wrapper that enforces 4:3 aspect ratio */
    [data-testid="stCameraInput"] video,
    [data-testid="stCameraInput"] img {
        width: 100% !important;
        max-width: 480px;
        height: min(75vw, 360px) !important;
        object-fit: cover !important;  
        object-position: center center;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 18px rgba(0,0,0,0.22);
        display: block;
    }

    /* Capture / shutter button — large & thumb-friendly */
    [data-testid="stCameraInput"] button {
        margin-top: 14px;
        width: min(80%, 280px) !important;
        min-height: 52px !important;
        font-size: 1.05rem !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 50px !important;
        background: #27ae60 !important;
        color: #fff !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(39,174,96,0.35) !important;
        transition: transform 0.1s, box-shadow 0.1s;
    }
    [data-testid="stCameraInput"] button:active {
        transform: scale(0.96);
        box-shadow: 0 2px 8px rgba(39,174,96,0.25) !important;
    }

    @media (max-width: 768px) {
        [data-testid="stCameraInput"] video,
        [data-testid="stCameraInput"] img {
            border-radius: 12px;
        }
        [data-testid="stCameraInput"] button {
            width: 90% !important;
            min-height: 56px !important;
            font-size: 1.1rem !important;
        }
    }

    /* ── Feedback buttons hover fix ───────────────────────────────────────── */
    div[data-testid="stButton"]:has(button[key*="fb_yes"]) button:hover,
    button[kind="secondary"]:hover:has(+ div) {
        background-color: rgba(39, 174, 96, 0.15) !important;
        border-color: #27ae60 !important;
        color: #27ae60 !important;
    }

    div[data-testid="stButton"]:has(button[key*="fb_no"]) button:hover {
        background-color: rgba(231, 76, 60, 0.13) !important;
        border-color: #e74c3c !important;
        color: #e74c3c !important;
    }

    /* ── Clear History button hover ────────────────────────────────────────── */
    html body button[aria-label="Clear History"]:hover {
        background-color: rgba(231, 76, 60, 0.13) !important;
        border-color:     #e74c3c !important;
        color:            #e74c3c !important;
        box-shadow:       0 0 0 2px rgba(231, 76, 60, 0.28) !important;
    }

</style>
""", unsafe_allow_html=True)


# DISEASE DATABASE


DISEASE_INFO = {
    "Eggplant Healthy Fruit": {
        "cause": "None - fruit is healthy",
        "symptom": "No disease detected; fruit surface appears normal with uniform color",
        "actions": [
            "Harvest at the correct maturity stage to avoid over-ripening.",
            "Handle fruits gently during harvest to prevent bruising.",
            "Store in cool, dry conditions away from direct sunlight.",
        ],
        "actions_fil": [
            "Anihin sa tamang yugto ng paghinog upang maiwasan ang sobrang paghinog.",
            "Pangalagaan ang mga bunga nang maingat habang pag-aani upang maiwasan ang pasa.",
            "Itago sa malamig at tuyong lugar na malayo sa direktang sikat ng araw.",
        ],
        "prevention": [
            "Continue regular field scouting every 3-5 days.",
            "Maintain balanced fertilization to support healthy fruit development.",
            "Ensure consistent soil moisture to prevent physiological disorders.",
            "Control insect pests early before they reach the fruiting stage.",
        ],
        "prevention_fil": [
            "Ipagpatuloy ang regular na pagsisiyasat sa bukid tuwing 3-5 araw.",
            "Panatilihin ang balanseng pataba upang suportahan ang malusog na pagbubunga.",
            "Tiyakin ang patuloy na kahalumigmigan ng lupa upang maiwasan ang mga karamdamang pisyolohikal.",
            "Kontrolin ang mga insektong peste nang maaga bago maabot ang yugto ng pagbubunga.",
        ],
        "severity": "None",
        "color": "#27ae60",
    },
    "Eggplant Phomopsis Blight": {
        "cause": "Phomopsis vexans (fungal pathogen)",
        "symptom": "Dark brown to black sunken lesions on fruit surface; may show concentric rings with pycnidia",
        "actions": [
            "Remove and destroy all infected fruits immediately.",
            "Apply mancozeb or copper-based fungicide to remaining fruits and foliage.",
            "Avoid overhead irrigation to reduce humidity around fruit.",
            "Improve canopy airflow through pruning of excess shoots.",
        ],
        "actions_fil": [
            "Alisin at sirain kaagad ang lahat ng nahawaang bunga.",
            "Mag-apply ng mancozeb o copper-based na fungicide sa natitirang bunga at dahon.",
            "Iwasan ang overhead na patubig upang bawasan ang kahalumigmigan sa paligid ng bunga.",
            "Pahusayin ang daloy ng hangin sa kalupaan sa pamamagitan ng pagputol ng labis na sanga.",
        ],
        "prevention": [
            "Use certified disease-free seeds or treated planting material.",
            "Practice strict crop rotation - avoid eggplant in the same field for 2-3 seasons.",
            "Apply preventive fungicide sprays during humid weather.",
            "Sanitize all field equipment to prevent spore transfer.",
        ],
        "prevention_fil": [
            "Gumamit ng sertipikadong binhi o pinagamot na materyal para sa pagtatanim.",
            "Magsagawa ng mahigpit na crop rotation - iwasan ang talong sa parehong bukid sa loob ng 2-3 panahon.",
            "Mag-apply ng preventive na fungicide spray sa panahon ng mataas na kahalumigmigan.",
            "Linisin ang lahat ng kagamitan sa bukid upang maiwasan ang paglipat ng spore.",
        ],
        "severity": "High",
        "color": "#8e44ad",
    },
    "Eggplant Shoot and Fruit Borer": {
        "cause": "Leucinodes orbonalis (insect pest - lepidopteran larva)",
        "symptom": "Small entry holes on fruit surface with frass; internal tunneling and rotting of flesh",
        "actions": [
            "Remove and destroy all bored fruits - do not leave them on the ground.",
            "Apply spinosad or emamectin benzoate targeting young larvae.",
            "Install pheromone traps to monitor adult moth populations.",
            "Prune bored shoots immediately and dispose of off-field.",
        ],
        "actions_fil": [
            "Alisin at sirain ang lahat ng nabutas na bunga - huwag iwanang nakakalat sa lupa.",
            "Mag-apply ng spinosad o emamectin benzoate na nakatutok sa batang larva.",
            "Mag-install ng pheromone trap upang subaybayan ang populasyon ng mga gamu-gamo.",
            "Putulin kaagad ang mga nabutas na sanga at itapon sa labas ng bukid.",
        ],
        "prevention": [
            "Use fine-mesh net bags over individual fruits during development.",
            "Plant at the start of the dry season to reduce borer pressure.",
            "Release egg parasitoids (Trichogramma spp.) as biocontrol agents.",
            "Maintain field sanitation - remove crop residues promptly after harvest.",
        ],
        "prevention_fil": [
            "Gumamit ng fine-mesh na supot sa bawat bunga habang lumalaki ito.",
            "Magtanim sa simula ng tag-araw upang mabawasan ang presyon ng borer.",
            "Maglabas ng egg parasitoid (Trichogramma spp.) bilang ahente ng biocontrol.",
            "Panatilihing malinis ang bukid - alisin agad ang mga natirang halaman pagkatapos ng ani.",
        ],
        "severity": "High",
        "color": "#c0392b",
    },
    "Eggplant Healthy Leaf": {
        "cause": "None - plant is healthy",
        "symptom": "No disease detected; foliage appears normal",
        "actions": [
            "Continue regular monitoring every 3-5 days.",
            "Maintain a balanced fertilization schedule (N-P-K).",
            "Ensure consistent, even soil moisture.",
        ],
        "actions_fil": [
            "Ipagpatuloy ang regular na pagmamasid tuwing 3-5 araw.",
            "Panatilihin ang balanseng iskedyul ng pagpapataba (N-P-K).",
            "Tiyakin ang patuloy at pantay na kahalumigmigan ng lupa.",
        ],
        "prevention": [
            "Keep up with preventive scouting routines.",
            "Monitor weather forecasts for sudden humidity changes.",
            "Maintain proper plant spacing for airflow.",
            "Remove weeds regularly to reduce pest harborage.",
        ],
        "prevention_fil": [
            "Ipagpatuloy ang mga preventive na gawi ng pagsisiyasat.",
            "Subaybayan ang ulat ng panahon para sa biglaang pagbabago ng kahalumigmigan.",
            "Panatilihin ang tamang espasyo ng halaman para sa daloy ng hangin.",
            "Regular na alisin ang damo upang mabawasan ang tirahan ng mga peste.",
        ],
        "severity": "None",
        "color": "#27ae60",
    },
    "Eggplant Insect Pest Disease": {
        "cause": "Various insect pests (aphids, thrips, mites, Leucinodes orbonalis)",
        "symptom": "Visible pest damage, holes, frass, stippling, or wilting on foliage caused by insect feeding",
        "actions": [
            "Identify the specific pest before choosing a control method.",
            "Apply appropriate insecticide (e.g., spinosad for borers, imidacloprid for sucking pests).",
            "Remove and destroy heavily infested leaves immediately.",
            "Install sticky yellow traps to monitor flying pest populations.",
        ],
        "actions_fil": [
            "Tukuyin ang tiyak na peste bago pumili ng paraan ng kontrol.",
            "Mag-apply ng angkop na insecticide (hal. spinosad para sa borer, imidacloprid para sa mga sumisipsip na peste).",
            "Alisin at sirain kaagad ang mga dahong lubhang nahawa.",
            "Mag-install ng malagkit na dilaw na bitag upang subaybayan ang mga lumilipad na peste.",
        ],
        "prevention": [
            "Use fine-mesh nets over seedbeds to exclude early-stage pests.",
            "Avoid dense planting - allow adequate airflow between plants.",
            "Introduce beneficial insects (e.g., ladybugs, Trichogramma wasps).",
            "Scout weekly and act at the first sign of pest activity.",
        ],
        "prevention_fil": [
            "Gumamit ng fine-mesh na lambat sa mga seedbed upang pigilan ang mga peste sa maagang yugto.",
            "Iwasan ang masikip na pagtatanim - payagan ang sapat na daloy ng hangin sa pagitan ng mga halaman.",
            "Magpakilala ng mga kapaki-pakinabang na insekto (hal. ladybug, Trichogramma wasp).",
            "Mag-scout linggu-linggo at kumilos sa unang palatandaan ng aktibidad ng peste.",
        ],
        "severity": "High",
        "color": "#e67e22",
    },
    "Eggplant Leaf Spot Disease": {
        "cause": "Fungal pathogens (Cercospora melongenae / Alternaria spp.)",
        "symptom": "Circular brown or grey lesions with defined margins on leaf surfaces; may have yellow halos",
        "actions": [
            "Apply chlorothalonil or mancozeb-based fungicide for lesion control.",
            "Remove and destroy heavily spotted leaves immediately.",
            "Avoid overhead irrigation to reduce leaf wetness.",
            "Apply organic mulch to prevent spore-laden soil splash onto leaves.",
        ],
        "actions_fil": [
            "Mag-apply ng chlorothalonil o mancozeb-based na fungicide para sa kontrol ng mga sugat.",
            "Alisin at sirain kaagad ang mga dahong may maraming batik.",
            "Iwasan ang overhead na patubig upang mabawasan ang pagbabasa ng dahon.",
            "Mag-apply ng organikong mulch upang maiwasan ang pagtagas ng lupa na may spore sa mga dahon.",
        ],
        "prevention": [
            "Avoid dense planting - allow adequate airflow between plants.",
            "Apply preventive fungicide sprays during humid or rainy weather.",
            "Scout weekly and act at the first sign of lesions.",
            "Practice crop rotation with non-solanaceous crops.",
        ],
        "prevention_fil": [
            "Iwasan ang masikip na pagtatanim - payagan ang sapat na daloy ng hangin sa pagitan ng mga halaman.",
            "Mag-apply ng preventive na fungicide spray sa panahon ng mataas na kahalumigmigan o ulan.",
            "Mag-scout linggu-linggo at kumilos sa unang palatandaan ng mga sugat.",
            "Magsagawa ng crop rotation gamit ang mga halaman na hindi solanaceous.",
        ],
        "severity": "Medium",
        "color": "#f39c12",
    },
    "Eggplant Wilt Disease": {
        "cause": "Ralstonia solanacearum (bacterial) or Fusarium oxysporum (fungal)",
        "symptom": "Sudden, progressive wilting of shoots and leaves despite adequate moisture",
        "actions": [
            "Remove and destroy wilted plants immediately; bag before carrying out.",
            "Drench surrounding soil with copper-based bactericide or biocontrol agents.",
            "Switch to drip irrigation - avoid overhead watering.",
            "Do not replant solanaceous crops in the same soil for at least 2 seasons.",
        ],
        "actions_fil": [
            "Alisin at sirain kaagad ang mga nalantang halaman; ilagay sa supot bago dalhin palabas.",
            "Basain ang kapaligid na lupa ng copper-based na bactericide o mga ahente ng biocontrol.",
            "Lumipat sa drip irrigation - iwasan ang overhead na pagdidilig.",
            "Huwag muling magtanim ng solanaceous na pananim sa parehong lupa sa loob ng hindi bababa sa 2 panahon.",
        ],
        "prevention": [
            "Use certified disease-free seedlings and resistant varieties.",
            "Practice crop rotation with non-solanaceous crops (e.g., corn, legumes).",
            "Disinfect tools between plants using 70% isopropyl alcohol.",
            "Improve soil drainage and avoid waterlogging.",
        ],
        "prevention_fil": [
            "Gumamit ng sertipikadong binhi at resistanteng uri ng halaman.",
            "Magsagawa ng crop rotation gamit ang mga halaman na hindi solanaceous (hal. mais, leguminosa).",
            "I-disinfect ang mga kagamitan sa pagitan ng mga halaman gamit ang 70% isopropyl alcohol.",
            "Pahusayin ang drainage ng lupa at iwasan ang pagbababad ng tubig.",
        ],
        "severity": "High",
        "color": "#d35400",
    },
}

CLASS_NAMES = list(DISEASE_INFO.keys())


# MODEL LOADER

@st.cache_resource
def load_model():
    model_path = "yieldy_model(matrix2).pth"

    if not os.path.exists(model_path):
        st.warning(
            f"Weight file `{model_path}` not found in the working directory. "
            "Running in **Demo Mode** with mock inference. "
            "Place `yieldy_model.pth` next to `app.py` to enable real inference.",
        )
        return None

    try:
        import torch
        import timm

        model = timm.create_model(
            "efficientnet_b2",
            pretrained=False,
            num_classes=len(CLASS_NAMES),
        )
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    except FileNotFoundError:
        st.error(
            "**Model file not found.** "
            f"Expected `{model_path}` in the same folder as `app.py`. "
            "Falling back to Demo Mode."
        )
        return None

    except RuntimeError as exc:
        st.error(
            "**Model architecture mismatch.** "
            "The checkpoint does not match EfficientNet-B2 with "
            f"{len(CLASS_NAMES)} output classes.  \n"
            f"Technical detail: `{exc}`  \n"
            "Falling back to Demo Mode."
        )
        return None

    except Exception as exc:
        import sys
        import importlib.util
        timm_spec = importlib.util.find_spec("timm")
        st.error(
            f"**Unexpected error loading model:** `{exc}`  \n"
            f"**Python executable:** `{sys.executable}`  \n"
            f"**timm visible to this Python:** `{bool(timm_spec)}`  \n"
            "Falling back to Demo Mode."
        )
        return None


# TTA INFERENCE

def _softmax(arr: np.ndarray) -> np.ndarray:
    e = np.exp(arr - arr.max())
    return e / e.sum()


def predict(image: Image.Image, model) -> tuple[str, float, dict]:
    if model is None:
        img_array = np.array(image.resize((260, 260))).astype(np.float32)
        seed = int(img_array.mean() * 100) % 2_147_483_647
        rng  = np.random.default_rng(seed)
        raw  = rng.dirichlet(np.ones(len(CLASS_NAMES)) * 0.5)
        top  = int(np.argmax(raw))
        raw[top] *= 3
        raw /= raw.sum()
        raw_scores = raw
    else:
        import torch
        from torchvision import transforms as T_tv

        base_tf = T_tv.Compose([
            T_tv.Resize((260, 260)),
            T_tv.ToTensor(),
            T_tv.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
        ])

        rgb = image.convert("RGB")
        variants = [
            rgb,
            ImageOps.mirror(rgb),
            ImageOps.flip(rgb),
        ]

        logit_accumulator = None
        with torch.no_grad():
            for variant in variants:
                tensor = base_tf(variant).unsqueeze(0)
                logits = model(tensor).squeeze(0).numpy()
                logit_accumulator = logits if logit_accumulator is None \
                    else logit_accumulator + logits

        avg_logits = logit_accumulator / len(variants)
        raw_scores = _softmax(avg_logits)

    predicted_idx   = int(np.argmax(raw_scores))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence      = float(raw_scores[predicted_idx])
    all_scores      = {CLASS_NAMES[i]: float(raw_scores[i]) for i in range(len(CLASS_NAMES))}

    return predicted_class, confidence, all_scores


# INVALID-IMAGE FILTER

def is_invalid_image(confidence: float, all_scores: dict) -> bool:
    if confidence < CONFIDENCE_THRESHOLD:
        return True
    probs = np.array(list(all_scores.values()), dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    entropy = float(-np.sum(probs * np.log(probs)))
    if entropy > ENTROPY_THRESHOLD:
        return True
    return False


# TWO-PHASE GATE

def run_gates(predicted_class: str, confidence: float, all_scores: dict):
    healthy_prob   = sum(all_scores.get(cls, 0.0) for cls in HEALTHY_CLASSES)
    disease_scores = {cls: s for cls, s in all_scores.items() if cls not in HEALTHY_CLASSES}
    top_disease      = max(disease_scores, key=disease_scores.get)
    top_disease_conf = disease_scores[top_disease]

    if predicted_class in HEALTHY_CLASSES and healthy_prob >= HEALTHY_GATE_THRESHOLD:
        return True, predicted_class, confidence, healthy_prob, top_disease, top_disease_conf

    return False, top_disease, top_disease_conf, healthy_prob, top_disease, top_disease_conf


# PDF REPORT

def generate_pdf_report(
    gate_class: str,
    gate_confidence: float,
    all_scores: dict,
    is_healthy: bool,
    healthy_prob: float,
    lang: str,
) -> bytes | None:
    try:
        from fpdf import FPDF, XPos, YPos
    except ImportError:
        return None

    info     = DISEASE_INFO[gate_class]
    severity = info["severity"]
    now      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    def _w():
        return pdf.w - pdf.l_margin - pdf.r_margin

    def _cell(h, txt, bold=False, size=10, color=(30, 30, 30), align="L"):
        pdf.set_font("Helvetica", "B" if bold else "", size)
        pdf.set_text_color(*color)
        pdf.set_x(pdf.l_margin)
        pdf.cell(_w(), h, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align=align)

    def _multi(h, txt, size=10, color=(30, 30, 30), style=""):
        pdf.set_font("Helvetica", style, size)
        pdf.set_text_color(*color)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(_w(), h, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    _cell(10, "YIELDLY - Eggplant Diagnosis Report", bold=True, size=20,
          color=(39, 174, 96), align="C")
    _cell(6, f"Generated: {now}  |  Language: {lang}", size=9,
          color=(100, 100, 100), align="C")
    pdf.ln(4)

    pdf.set_draw_color(39, 174, 96)
    pdf.set_line_width(0.5)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)

    phase_tag = "Phase 1 - Healthy" if is_healthy else "Phase 2 - Disease Detected"
    _cell(8, phase_tag, bold=True, size=13)
    _cell(9, gate_class, bold=True, size=15, color=(50, 50, 50))
    _multi(6, f"Cause: {info['cause']}", size=10, color=(100, 100, 100), style="I")
    pdf.ln(2)

    sev_colors = {"High": (192, 57, 43), "Medium": (243, 156, 18), "None": (39, 174, 96)}
    r, g, b = sev_colors.get(severity, (50, 50, 50))
    _cell(6, f"Severity: {severity}  |  Confidence: {gate_confidence * 100:.1f}%",
          bold=True, size=10, color=(r, g, b))

    if not is_healthy:
        _cell(5, f"Healthy probability: {healthy_prob * 100:.1f}%",
              size=9, color=(100, 100, 100))
    pdf.ln(3)

    _cell(6, "Observation / Primary Symptom:", bold=True, size=10)
    _multi(5, info["symptom"])
    pdf.ln(2)

    _cell(6, "Recommended Actions:", bold=True, size=10)
    for i, action in enumerate(info["actions"], 1):
        _multi(5, f"{i}. {action}")
    pdf.ln(2)

    _cell(6, "Prevention Tips:", bold=True, size=10)
    for tip in info["prevention"]:
        _multi(5, f"- {tip}")
    pdf.ln(2)

    _cell(6, "Confidence Breakdown (All Classes):", bold=True, size=10)
    for cls, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
        _multi(5, f"{cls}   {score * 100:.1f}%", size=9)
    pdf.ln(3)

    pdf.set_draw_color(180, 180, 180)
    pdf.set_line_width(0.3)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)
    _multi(4,
           "This report is for preliminary reference only and does not replace the advice "
           "of a qualified agronomist or plant pathologist. YIELDLY v1.0",
           size=8, color=(130, 130, 130), style="I")

    return bytes(pdf.output())


# PDF SUMMARY REPORT

def generate_summary_pdf_report(history: list, lang: str) -> bytes | None:
    if not history:
        return None

    try:
        from fpdf import FPDF, XPos, YPos
    except ImportError:
        return None

    now        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    total      = len(history)
    n_healthy  = sum(1 for r in history if r["is_healthy"])
    n_disease  = total - n_healthy

    sev_colors = {
        "High":   (192,  57,  43),
        "Medium": (243, 156,  18),
        "None":   ( 39, 174,  96),
    }

    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    def _w():
        return pdf.w - pdf.l_margin - pdf.r_margin

    def _cell(h, txt, bold=False, size=10, color=(30, 30, 30), align="L"):
        pdf.set_font("Helvetica", "B" if bold else "", size)
        pdf.set_text_color(*color)
        pdf.set_x(pdf.l_margin)
        pdf.cell(_w(), h, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align=align)

    def _multi(h, txt, size=10, color=(30, 30, 30), style=""):
        pdf.set_font("Helvetica", style, size)
        pdf.set_text_color(*color)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(_w(), h, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def _hline(r=39, g=174, b=96, lw=0.5):
        pdf.set_draw_color(r, g, b)
        pdf.set_line_width(lw)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(3)

    _cell(11, tr("pdf_summary_title", lang), bold=True, size=18,
          color=(39, 174, 96), align="C")
    _cell(6,  f"Generated: {now}  |  Language: {lang}", size=9,
          color=(110, 110, 110), align="C")
    pdf.ln(3)
    _hline(39, 174, 96, lw=0.8)

    stats_y = pdf.get_y()
    pdf.set_fill_color(245, 255, 248)
    pdf.rect(pdf.l_margin, stats_y, _w(), 22, style="F")

    pdf.set_y(stats_y + 4)
    col_w = _w() / 3
    stats = [
        (tr("pdf_total_scans",   lang), str(total)),
        (tr("pdf_healthy_count", lang), str(n_healthy)),
        (tr("pdf_disease_count", lang), str(n_disease)),
    ]
    stat_colors = [(39, 174, 96), (39, 174, 96), (192, 57, 43)]
    x_cursor = pdf.l_margin
    for (label, value), col_color in zip(stats, stat_colors):
        pdf.set_x(x_cursor)
        pdf.set_font("Helvetica", "B", 15)
        pdf.set_text_color(*col_color)
        pdf.cell(col_w, 8, value, align="C")
        x_cursor += col_w

    pdf.ln(8)
    x_cursor = pdf.l_margin
    for label, _ in stats:
        pdf.set_x(x_cursor)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(col_w, 5, label, align="C")
        x_cursor += col_w

    pdf.ln(9)
    _hline(39, 174, 96, lw=0.5)

    for idx, rec in enumerate(history, start=1):
        info      = DISEASE_INFO.get(rec["diagnosis"], {})
        severity  = rec.get("severity", "None")
        sev_rgb   = sev_colors.get(severity, (80, 80, 80))
        phase_tag = "Phase 1 - Healthy" if rec["is_healthy"] else "Phase 2 - Disease Detected"
        conf_pct  = f"{rec['confidence'] * 100:.1f}%"

        scan_label = tr("pdf_scan_label", lang)
        _cell(7,
              f"{scan_label} {idx}  |  {rec['filename']}",
              bold=True, size=11, color=(40, 40, 40))

        _cell(5, rec["timestamp"], size=8, color=(130, 130, 130))
        pdf.ln(1)

        _cell(6, phase_tag, bold=False, size=9, color=(100, 100, 100))
        _cell(8, rec["diagnosis"], bold=True, size=13, color=(30, 30, 30))

        if info.get("cause"):
            _multi(5, f"Cause: {info['cause']}", size=9,
                   color=(110, 110, 110), style="I")
        pdf.ln(1)

        r_s, g_s, b_s = sev_rgb
        _cell(6,
              f"Confidence: {conf_pct}    |    Severity: {severity}",
              bold=True, size=10, color=(r_s, g_s, b_s))
        pdf.ln(2)

        if info.get("symptom"):
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(50, 50, 50)
            pdf.set_x(pdf.l_margin)
            pdf.cell(_w(), 5, "Primary Symptom:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            _multi(4.5, info["symptom"], size=9, color=(60, 60, 60))
            pdf.ln(1)

        if info.get("actions"):
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(50, 50, 50)
            pdf.set_x(pdf.l_margin)
            pdf.cell(_w(), 5, "Recommended Actions:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            for action_num, action in enumerate(info["actions"], 1):
                _multi(4.5, f"  {action_num}. {action}", size=9, color=(60, 60, 60))
            pdf.ln(1)

        if info.get("prevention"):
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(50, 50, 50)
            pdf.set_x(pdf.l_margin)
            pdf.cell(_w(), 5, "Prevention Tips:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            for tip in info["prevention"]:
                _multi(4.5, f"  - {tip}", size=9, color=(60, 60, 60))

        pdf.ln(3)

        if idx < total:
            _hline(180, 180, 180, lw=0.3)
            pdf.ln(1)

    pdf.ln(4)
    _hline(180, 180, 180, lw=0.3)
    _multi(4,
           "This report is for preliminary reference only and does not replace the advice "
           "of a qualified agronomist or plant pathologist. YIELDLY v1.0",
           size=8, color=(140, 140, 140), style="I")

    return bytes(pdf.output())


# UI HELPERS

def _crop_to_ratio(img: Image.Image, target_w: int = 4, target_h: int = 3) -> Image.Image:
    src_w, src_h = img.size
    target_ratio = target_w / target_h
    src_ratio    = src_w   / src_h

    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        left  = (src_w - new_w) // 2
        return img.crop((left, 0, left + new_w, src_h))
    elif src_ratio < target_ratio:
        new_h = int(src_w / target_ratio)
        top   = (src_h - new_h) // 2
        return img.crop((0, top, src_w, top + new_h))
    else:
        return img


def severity_badge(severity: str, color: str) -> str:
    return (
        f'<span style="background:{color}22; color:{color}; border:1px solid {color}; '
        f'padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; '
        f'font-family:Inter,sans-serif;">Severity: {severity}</span>'
    )


def confidence_color(conf: float) -> str:
    if conf >= 0.80:
        return "#27ae60"
    elif conf >= 0.55:
        return "#e67e22"
    return "#e74c3c"


def render_severity_banner(severity: str, lang: str):
    if severity == "High":
        css_class = "banner-high"
        msg = tr("banner_high", lang)
    elif severity == "Medium":
        css_class = "banner-medium"
        msg = tr("banner_medium", lang)
    else:
        css_class = "banner-none"
        msg = tr("banner_none", lang)

    st.markdown(
        f'<div class="{css_class}">'
        f'<span style="font-family:Inter,sans-serif; font-size:0.9rem; font-weight:600;">'
        f'{msg}</span></div>',
        unsafe_allow_html=True,
    )


def image_hash(img: Image.Image) -> str:
    arr = np.array(img.resize((32, 32))).tobytes()
    return hashlib.md5(arr).hexdigest()[:10]


# SESSION STATE INITIALISATION

def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}
    if "language" not in st.session_state:
        st.session_state.language = "English"
    if "current_img_idx" not in st.session_state:
        st.session_state.current_img_idx = 0
    if "last_upload_count" not in st.session_state:
        st.session_state.last_upload_count = 0
    if "history_search_query" not in st.session_state:
        st.session_state.history_search_query = ""


# SINGLE IMAGE RESULT RENDERER

def render_result(image: Image.Image, filename: str, model, lang: str):
    
    with st.container(border=True):
        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            st.image(image, caption=tr("uploaded_image", lang), use_container_width=True)
            st.caption(filename)

        with col_res:
            st.markdown(
                f"<p style='font-family:Inter,sans-serif; font-size:0.95rem; font-weight:500; color:#ffffff !important; color:gray;'>"
                f"{tr('analyzing', lang)}</p>",
                unsafe_allow_html=True,
            )
            progress = st.progress(0)
            for i in range(1, 101):
                time.sleep(0.006)
                progress.progress(i)

            predicted_class, confidence, all_scores = predict(image, model)

            if is_invalid_image(confidence, all_scores):
                progress.empty()
                st.warning(
                    f"**{tr('invalid_image_title', lang)}**\n\n"
                    f"{tr('invalid_image_body', lang)}",
                )
                st.caption(
                    f"Model confidence: {confidence * 100:.1f}% "
                    f"(minimum required: {CONFIDENCE_THRESHOLD * 100:.0f}%)"
                )
                return

            is_healthy, gate_class, gate_confidence, healthy_prob, top_disease, top_disease_conf = \
                run_gates(predicted_class, confidence, all_scores)

            info       = DISEASE_INFO[gate_class]
            conf_color = confidence_color(gate_confidence)

            progress.empty()

            if is_healthy:
                st.markdown(f"""
                <div style="border:1px solid {info['color']}; border-radius:12px;
                            padding:16px 20px; background:{info['color']}11;">
                    <div style="font-family:Poppins,sans-serif; font-size:0.88rem;
                                font-weight:600; color:#27ae60; margin-bottom:6px;">
                        {tr('phase1_passed', lang)}
                    </div>
                    <div style="font-family:Poppins,sans-serif; font-size:1.15rem;
                                font-weight:700; color:{info['color']};">{gate_class}</div>
                    <div style="font-family:Inter,sans-serif; font-size:0.78rem;
                                color:gray; margin:4px 0 8px 0;"><i>{info['cause']}</i></div>
                    <div style="font-family:Inter,sans-serif; margin-top:12px; font-size:0.84rem;">
                        <b>{tr('observation', lang)}:</b> {info['symptom']}
                    </div>
                    <div style="margin-top:8px;">
                        <span style="font-family:Poppins,sans-serif; font-size:1.35rem;
                                     font-weight:700; color:{conf_color};">
                            {gate_confidence * 100:.1f}%
                        </span>
                        <span style="font-family:Inter,sans-serif; font-size:0.82rem; color:gray;">
                            {tr('confidence_label', lang)} (TTA)
                        </span>
                    </div>
                    <div style="font-family:Inter,sans-serif; margin-top:10px;
                                font-size:0.73rem; color:gray;">
                        {tr('healthy_prob', lang)}: {healthy_prob * 100:.1f}%
                        (threshold: {HEALTHY_GATE_THRESHOLD * 100:.0f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div style="border:1px solid #e74c3c; border-radius:12px;
                            padding:10px 16px; background:#e74c3c11; margin-bottom:10px;">
                    <span style="font-family:Inter,sans-serif; font-size:0.86rem;
                                 color:#e74c3c; font-weight:600;">
                        {tr('phase1_fail', lang)}
                    </span><br>
                    <span style="font-family:Inter,sans-serif; font-size:0.76rem; color:gray;">
                        {tr('healthy_prob', lang)}: {healthy_prob * 100:.1f}%
                    </span>
                </div>
                <div style="border:1px solid {info['color']}; border-radius:12px;
                            padding:16px 20px; background:{info['color']}11;">
                    <div style="font-family:Poppins,sans-serif; font-size:0.88rem;
                                font-weight:600; color:{info['color']}; margin-bottom:6px;">
                        {tr('phase2_label', lang)}
                    </div>
                    <div style="font-family:Poppins,sans-serif; font-size:1.15rem;
                                font-weight:700; color:{info['color']};">{gate_class}</div>
                    <div style="font-family:Inter,sans-serif; font-size:0.78rem;
                                color:gray; margin:4px 0 8px 0;"><i>{info['cause']}</i></div>
                    {severity_badge(info['severity'], info['color'])}
                    <div style="font-family:Inter,sans-serif; margin-top:12px; font-size:0.84rem;">
                        <b>{tr('primary_symptom', lang)}:</b> {info['symptom']}
                    </div>
                    <div style="margin-top:8px;">
                        <span style="font-family:Poppins,sans-serif; font-size:1.35rem;
                                     font-weight:700; color:{conf_color};">
                            {gate_confidence * 100:.1f}%
                        </span>
                        <span style="font-family:Inter,sans-serif; font-size:0.82rem; color:gray;">
                            {tr('confidence_label', lang)} (TTA)
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        render_severity_banner(info["severity"], lang)

    col3, col4 = st.columns(2, gap="large")
    with col3:
        with st.container(border=True):
            st.markdown(
                f"<p style='font-family:Poppins,sans-serif; font-weight:600; font-size:0.92rem; color:#ffffff; display:flex; align-items:center; gap:6px;'>"
                f"<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='#ffffff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='flex-shrink:0;'><path d='M14 9.536V7a4 4 0 0 1 4-4h1.5a.5.5 0 0 1 .5.5V5a4 4 0 0 1-4 4 4 4 0 0 0-4 4c0 2 1 3 1 5a5 5 0 0 1-1 3'/><path d='M4 9a5 5 0 0 1 8 4 5 5 0 0 1-8-4'/><path d='M5 21h14'/></svg>"
                f" {tr('recommended_actions', lang)}</p>",
                unsafe_allow_html=True,
            )
            actions_key = "actions_fil" if lang == "Filipino" else "actions"
            actions_html = "".join(
                f"<li style='margin-bottom:6px; color:#ffffff;'>{action}</li>"
                for action in info[actions_key]
            )
            st.markdown(
                f"<ul style='color:#ffffff; font-family:Inter,sans-serif; font-size:0.95rem; font-weight:500; color:#ffffff !important; "
                f"line-height:1.6; padding-left:1.3rem; margin:4px 0 0 0;'>"
                f"{actions_html}</ul>",
                unsafe_allow_html=True,
            )

    with col4:
        with st.container(border=True):
            st.markdown(
                f"<p style='font-family:Poppins,sans-serif; font-weight:600; font-size:0.92rem; color:#ffffff; display:flex; align-items:center; gap:6px;'>"
                f"<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='#ffffff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='flex-shrink:0;'><path d='M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z'/></svg>"
                f" {tr('prevention_tips', lang)}</p>",
                unsafe_allow_html=True,
            )
            prevention_key = "prevention_fil" if lang == "Filipino" else "prevention"
            prevention_html = "".join(
                f"<li style='margin-bottom:6px; color:#ffffff;'>{tip}</li>"
                for tip in info[prevention_key]
            )
            st.markdown(
                f"<ul style='color:#ffffff; font-family:Inter,sans-serif; font-size:0.95rem; font-weight:500; color:#ffffff !important; "
                f"line-height:1.6; padding-left:1.3rem; margin:4px 0 0 0;'>"
                f"{prevention_html}</ul>",
                unsafe_allow_html=True,
            )

    st.divider()

    with st.expander(f"{tr('confidence_breakdown', lang)} & Export Options"):
        st.caption(tr("tta_note", lang))

        for cls, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
            bar_color   = DISEASE_INFO[cls]["color"]
            bar_pct     = score * 100
            is_selected = cls == gate_class
            lbl_style   = "font-weight:700;" if is_selected else "color:gray;"
            gate_tag    = " [selected]" if is_selected else ""
            st.markdown(f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                    <span style="font-family:Inter,sans-serif; font-size:0.81rem; {lbl_style}">
                        {cls}{gate_tag}
                    </span>
                    <span style="font-family:Inter,sans-serif; font-size:0.81rem; {lbl_style}">
                        {bar_pct:.1f}%
                    </span>
                </div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar-fill" style="width:{bar_pct}%; background:{bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

    st.divider()

    img_key  = image_hash(image)
    feedback = st.session_state.feedback.get(img_key)

    st.markdown(
        f"<p style='font-family:Inter,sans-serif; font-size:0.88rem; font-weight:600; text-align:center;'>"
        f"{tr('feedback_prompt', lang)}</p>",
        unsafe_allow_html=True,
    )
    
    if feedback:
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:8px; background:#e8f8ef; border:1px solid #27ae60; "
            f"border-radius:8px; padding:10px 14px; font-family:Inter,sans-serif; font-size:0.9rem; color:#1a7a40;'>"
            f"<svg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='#27ae60' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='flex-shrink:0;'>"
            f"<path d='M18 6 7 17l-5-5'/><path d='m22 10-7.5 7.5L13 16'/></svg>"
            f" {tr('feedback_thanks', lang)}</div>",
            unsafe_allow_html=True,
        )
    else:
        fb_col1, fb_col2, fb_col3, fb_col4 = st.columns([1, 1, 1, 1])
        if fb_col2.button(
            tr("feedback_yes", lang),
            icon=":material/thumb_up:",
            key=f"fb_yes_{img_key}",
            use_container_width=True,
        ):
            st.session_state.feedback[img_key] = "yes"
            st.rerun()
        if fb_col3.button(
            tr("feedback_no", lang),
            icon=":material/thumb_down:",
            key=f"fb_no_{img_key}",
            use_container_width=True,
        ):
            st.session_state.feedback[img_key] = "no"
            st.rerun()


    record = {
        "timestamp":  datetime.datetime.now().strftime("%H:%M:%S"),
        "filename":   filename,
        "diagnosis":  gate_class,
        "confidence": gate_confidence,
        "severity":   info["severity"],
        "is_healthy": is_healthy,
        "img_hash":   img_key,
    }
    existing_keys = {h.get("img_hash") for h in st.session_state.history}
    if img_key not in existing_keys:
        st.session_state.history.insert(0, record)


# SESSION HISTORY PANEL  (rendered in main content)

def render_history_panel(lang: str):
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander(f"{tr('history_title', lang)}", icon=":material/history:", expanded=False):
        history = st.session_state.history

        if not history:
            st.info(tr("pdf_no_history", lang))
            return

        total      = len(history)
        n_healthy  = sum(1 for r in history if r["is_healthy"])
        n_disease  = total - n_healthy
        sev_counts = {"High": 0, "Medium": 0, "None": 0}
        
        for r in history:
            sev_counts[r["severity"]] = sev_counts.get(r["severity"], 0) + 1

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Total Diagnoses", total)
        col_b.metric("Healthy", n_healthy)
        col_c.metric("Disease Detected", n_disease)
        col_d.metric("High Severity", sev_counts.get("High", 0))

        st.markdown("")

        filtered_history = history

        export_col, clear_col = st.columns([3, 1])
        with export_col:
            with st.spinner("Building PDF..."):
                summary_pdf = generate_summary_pdf_report(history, lang)
            if summary_pdf:
                st.download_button(
                    label=tr("export_pdf_summary", lang),
                    data=summary_pdf,
                    file_name=tr("pdf_summary_filename", lang),
                    mime="application/pdf",
                    key="pdf_summary_export",
                    use_container_width=True,
                )

        with clear_col:
            if st.button(
                "Clear History",
                icon=":material/delete:",
                key="hist_clear_main",
                help="Clear all session history",
                use_container_width=True,
            ):
                st.session_state.history.clear()
                st.session_state.feedback.clear()
                st.rerun()

        st.markdown("")

        sev_color_map = {"High": "#e74c3c", "Medium": "#f39c12", "None": "#27ae60"}

        if not filtered_history:
            st.caption("No diagnoses recorded yet.")
        else:
            for i, rec in enumerate(filtered_history):
                color     = sev_color_map.get(rec["severity"], "#999")
                fb        = st.session_state.feedback.get(rec["img_hash"], "")
                fb_text   = " (Feedback: Accurate)" if fb == "yes" else (" (Feedback: Inaccurate)" if fb == "no" else "")
                phase_tag = "Healthy" if rec["is_healthy"] else "Disease"

                row_left, row_right = st.columns([5, 1])
                with row_left:
                    st.markdown(
                        f"<div class='hist-item'>"
                        f"<span style='color:{color}; font-weight:700;'>[{rec['severity']}]</span> "
                        f"<b>{rec['diagnosis']}</b> "
                        f"<span style='color:gray; font-size:0.75rem;'>({phase_tag})</span>"
                        f"<br>"
                        f"<span style='color:#aaa; font-size:0.75rem;'>"
                        f"{rec['timestamp']} &nbsp;|&nbsp; {rec['filename'][:30]} "
                        f"&nbsp;|&nbsp; {rec['confidence']*100:.0f}% confidence"
                        f"{fb_text}"
                        f"</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with row_right:
                    orig_idx = st.session_state.history.index(rec)
                    if st.button(tr("history_remove", lang), key=f"hist_rm_{i}_{rec['img_hash']}"):
                        st.session_state.history.pop(orig_idx)
                        st.rerun()


# MAIN APP

def main():
    init_session_state()

    with st.sidebar:
        import base64 as _b64
        with open("yieldlylogo111.png", "rb") as _f:
            _logo_b64 = _b64.b64encode(_f.read()).decode()
        st.markdown(
            f"<img src='data:image/png;base64,{_logo_b64}' alt='Yieldly' "
            f"style='height:60px; width:auto; object-fit:contain; margin-bottom:2px;' />",
            unsafe_allow_html=True,
        )
        st.caption("A crop disease detection tool for Filipino Eggplant Farmers.")
        st.divider()

        st.markdown(
            f"<p style='font-family:Inter,sans-serif; font-size:0.95rem; font-weight:500; color:#ffffff !important; "
            f"font-weight:600;'>{tr('sidebar_language', 'English')}</p>",
            unsafe_allow_html=True,
        )
        selected_lang = st.selectbox(
            "Language",
            options=LANG_OPTIONS,
            index=LANG_OPTIONS.index(st.session_state.language),
            label_visibility="collapsed",
        )
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            st.rerun()

        lang = st.session_state.language
        st.divider()

        with st.expander(" Model Info & Logic"):
            st.markdown(f"**{tr('sidebar_classes', lang)}**")
            for cls in CLASS_NAMES:
                color = DISEASE_INFO[cls]["color"]
                st.markdown(
                    f"<span style='color:{color}'>&#9679;</span> {cls}",
                    unsafe_allow_html=True,
                )
            st.divider()
            st.markdown(f"**{tr('sidebar_logic', lang)}**")
            st.markdown("**Phase 1** — Health Check (>=80% healthy confidence)")
            st.markdown("**Phase 2** — Disease Detector (if Phase 1 fails)")
            st.divider()
            st.markdown(f"**{tr('sidebar_model', lang)}**")
            st.markdown("**Target Crop:** Eggplant *(Solanum Melongena)*")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-family:Poppins,sans-serif; font-weight:600; font-size:0.9rem;'>"
            f"{tr('history_title', lang)}</p>",
            unsafe_allow_html=True,
        )
        if not st.session_state.history:
            st.caption(tr("history_empty", lang))
        else:
            st.caption(f"{len(st.session_state.history)} diagnosis(es) this session")
            for rec in st.session_state.history[:5]:   
                sev_color = {"High": "#e74c3c", "Medium": "#f39c12", "None": "#27ae60"}.get(
                    rec["severity"], "#999"
                )
                st.markdown(
                    f"<div class='hist-item'>"
                    f"<span style='color:{sev_color}; font-weight:700;'>{rec['severity']}</span> "
                    f"— <b>{rec['diagnosis'][:28]}</b><br>"
                    f"<span style='color:gray;'>{rec['timestamp']} · {rec['confidence']*100:.0f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            if len(st.session_state.history) > 5:
                st.caption(f"...and {len(st.session_state.history) - 5} more (see below)")

        st.caption("YIELDLY V1.0")

    lang = st.session_state.language

    import base64 as _b64
    with open("yieldlylogo111.png", "rb") as _f:
        _logo_b64 = _b64.b64encode(_f.read()).decode()
    st.markdown(f"""
    <div style="text-align:center; padding:0 0 1.5rem 0;">
        <img src="data:image/png;base64,{_logo_b64}"
             alt="Yieldly"
             style="height:110px; width:auto; object-fit:contain; margin-bottom:4px;" />
        <p style="font-family:Inter,sans-serif; color:gray; font-size:0.95rem; margin-top:6px;">
            {tr('app_subtitle', lang)}
        </p>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.info(tr("demo_mode", lang), icon="ℹ️")

    with st.container(border=True):
        st.markdown(
            f"<p style='font-family:Poppins,sans-serif; font-weight:600; font-size:1rem; margin-bottom:0; display:flex; align-items:center; gap:7px;'>"
            f"<svg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='flex-shrink:0; vertical-align:middle;'><path d='M13.997 4a2 2 0 0 1 1.76 1.05l.486.9A2 2 0 0 0 18.003 7H20a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h1.997a2 2 0 0 0 1.759-1.048l.489-.904A2 2 0 0 1 10.004 4z'/><circle cx='12' cy='13' r='3'/></svg>"
            f" {tr('upload_header', lang)}</p>",
            unsafe_allow_html=True,
        )

        tab_upload_label  = tr("tab_upload", lang)
        tab_camera_label  = tr("tab_quick_capture", lang)
        tab_upload, tab_camera = st.tabs([tab_upload_label, tab_camera_label])

        images_to_process: list[tuple[Image.Image, str]] = []

        with tab_upload:
            st.caption(f"{tr('upload_instruction', lang)} {tr('supported_formats', lang)}")
            uploaded_files = st.file_uploader(
                label="Choose image(s)",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
                accept_multiple_files=True,
            )

            if uploaded_files:
                for uf in uploaded_files:
                    try:
                        pil_img = Image.open(uf).convert("RGB")
                        images_to_process.append((pil_img, uf.name))
                    except Exception as e:
                        st.error(f"Could not open `{uf.name}`: {e}")

                if len(uploaded_files) != st.session_state.last_upload_count:
                    st.session_state.current_img_idx = 0
                    st.session_state.last_upload_count = len(uploaded_files)

        with tab_camera:
            st.caption(tr("camera_instruction", lang))
            camera_photo = st.camera_input(
                label=tr("take_a_photo", lang),
                label_visibility="visible",
            )

            if camera_photo is not None:
                try:
                    cam_img_raw = Image.open(camera_photo).convert("RGB")
                    cam_img = _crop_to_ratio(cam_img_raw, target_w=4, target_h=3)
                    images_to_process.append((cam_img, "camera_capture.jpg"))
                    st.success(tr("camera_photo_ready", lang))
                    if st.session_state.last_upload_count != 1:
                        st.session_state.current_img_idx = 0
                        st.session_state.last_upload_count = 1
                except Exception as e:
                    st.error(f"Could not process camera photo: {e}")

    if images_to_process:
        total_imgs = len(images_to_process)
        st.markdown("<br>", unsafe_allow_html=True)

        if total_imgs > 1:
            idx = min(st.session_state.current_img_idx, total_imgs - 1)
            st.session_state.current_img_idx = idx

            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
            with nav_col1:
                if st.button(
                    f"← {tr('nav_prev', lang)}",
                    disabled=(idx == 0),
                    key="nav_prev_btn",
                    use_container_width=True
                ):
                    st.session_state.current_img_idx -= 1
                    st.rerun()
            with nav_col2:
                label = (
                    tr("nav_image_of", lang)
                    .replace("{current}", str(idx + 1))
                    .replace("{total}", str(total_imgs))
                )
                st.markdown(
                    f"<div style='text-align:center; font-family:Inter,sans-serif; "
                    f"font-size:0.88rem; color:gray; padding-top:6px;'>{label}</div>",
                    unsafe_allow_html=True,
                )
            with nav_col3:
                if st.button(
                    f"{tr('nav_next', lang)} →",
                    disabled=(idx == total_imgs - 1),
                    key="nav_next_btn",
                    use_container_width=True
                ):
                    st.session_state.current_img_idx += 1
                    st.rerun()

            st.markdown(
                f"<h4 style='font-family:Poppins,sans-serif; margin-top:0.5rem; text-align:center; color:gray;'>"
                f"{images_to_process[idx][1]}</h4>",
                unsafe_allow_html=True,
            )
            render_result(images_to_process[idx][0], images_to_process[idx][1], model, lang)

        else:
            render_result(images_to_process[0][0], images_to_process[0][1], model, lang)

    render_history_panel(lang)

    if images_to_process:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:flex; align-items:flex-start; gap:10px; background:#fff8e1; border:1px solid #f39c12; "
            f"border-left:5px solid #f39c12; border-radius:8px; padding:12px 16px; "
            f"font-family:Inter,sans-serif; font-size:0.88rem; color:#7a5800;'>"
            f"<svg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='#f39c12' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='flex-shrink:0; margin-top:1px;'>"
            f"<path d='m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3'/>"
            f"<path d='M12 9v4'/><path d='M12 17h.01'/></svg>"
            f"<span>{tr('disclaimer', lang)}</span></div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()