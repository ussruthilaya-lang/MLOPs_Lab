import os
from pathlib import Path
import requests
import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

# ---------------- Backend URL ----------------
FASTAPI_BACKEND_ENDPOINT = os.getenv(
    "FASTAPI_URL", "https://flower-api-eqkifc3wnq-uc.a.run.app"
)

# ---------------- Species → Family mapping (fallback) ----------------
# Backend v2 already returns family; this is only for backward compatibility.
SPECIES_TO_FAMILY = {
    "setosa": "iris",
    "versicolor": "iris",
    "virginica": "iris",
    "lily_asiatic": "lily",
    "lily_oriental": "lily",
}

# ---------------- Asset paths ----------------
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets" / "flowers"

FAMILY_IMAGE_MAP = {
    "iris": ASSETS_DIR / "iris" / "family.jpg",
    "lily": ASSETS_DIR / "lily" / "family.jpg",
}

SPECIES_IMAGE_MAP = {
    "setosa": ASSETS_DIR / "iris" / "Iris_Setosa.png",
    "versicolor": ASSETS_DIR / "iris" / "Iris_Versicolor.png",
    "virginica": ASSETS_DIR / "iris" / "Iris_Virginica.png",
    "lily_asiatic": ASSETS_DIR / "lily" / "asiatic.png",
    "lily_oriental": ASSETS_DIR / "lily" / "oriental.png",
}

# ---------------- Themes ----------------
CONF_THEME = {
    "good": "linear-gradient(135deg, #134E5E, #71B280)",
    "medium": "linear-gradient(135deg, #3A5BA0, #6B8CFF)",
    "bad": "linear-gradient(135deg, #7A1E1E, #B23A3A)",
    "offline": "linear-gradient(135deg, #2C2C2C, #4A4A4A)",
}

def normalize_species_name(species: str) -> str:
    """
    Normalize backend species names to match UI asset keys.
    Examples:
    - Iris-setosa -> setosa
    - Iris-versicolor -> versicolor
    - lily_asiatic -> lily_asiatic
    """
    if species.lower().startswith("iris-"):
        return species.split("-", 1)[1].lower()
    return species.lower()


def apply_background(theme_css: str) -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {theme_css};
            color: #111111;
        }}
        h1, h2, h3, h4 {{ color: #0F172A; }}
        p, span, li, label {{ color: #111111; }}
        .stAlert {{ color: #111111; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def backend_is_alive(url: str) -> bool:
    try:
        r = requests.get(f"{url}/", timeout=2)
        return r.ok
    except Exception:
        return False


def safe_image(path: Path, caption: str) -> None:
    if path and path.is_file():
        st.image(str(path), caption=caption, width=320)
    else:
        st.info(f"Image missing: {caption}")


def classify_confidence(conf: float) -> str:
    if conf >= 0.70:
        return "good"
    if conf >= 0.50:
        return "medium"
    return "bad"


def run():
    st.set_page_config(
        page_title="Flower Species Prediction Dashboard",
        page_icon="🌸",
        layout="wide",
    )

    # ---------------- OFFLINE SCREEN ----------------
    if not backend_is_alive(FASTAPI_BACKEND_ENDPOINT):
        apply_background(CONF_THEME["offline"])
        st.title("🌸 Flower Species Prediction Dashboard")
        st.warning("Backend is offline. Predictions are disabled.")

        st.markdown(
            f"""
            **Backend URL:** `{FASTAPI_BACKEND_ENDPOINT}`

            **Fix checklist:**
            - Start FastAPI (`uvicorn main:app --reload`)
            - Confirm it opens at `{FASTAPI_BACKEND_ENDPOINT}/`
            - Refresh this page
            """
        )
        return

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.success("Backend online ✅")
        st.caption(f"Backend URL: {FASTAPI_BACKEND_ENDPOINT}")

        st.header("🌼 Input Features")

        sepal_length = st.slider("Sepal Length (cm)", 4.3, 9.0, 5.1, 0.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
        petal_length = st.slider("Petal Length (cm)", 1.0, 9.5, 1.4, 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 3.8, 0.2, 0.1)

        predict_button = st.button("Predict", type="primary")

    # ---------------- MAIN ----------------
    st.title("🌸 Flower Species Prediction Dashboard")
    st.write(
        "This dashboard visualizes **model predictions with confidence**. "
        "All learning and inference live in the backend API."
    )

    if not predict_button:
        apply_background(CONF_THEME["offline"])
        st.info("Set feature values in the sidebar and click **Predict**.")
        return

    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    try:
        with st.spinner("Running inference..."):
            response = requests.post(
                f"{FASTAPI_BACKEND_ENDPOINT}/predict",
                json=payload,
                timeout=5,
            )

        if response.status_code != 200:
            apply_background(CONF_THEME["offline"])
            st.error(f"Backend error: {response.status_code}")
            st.code(response.text)
            return

        result = response.json()

        raw_species = result.get("species", "unknown")
        species = normalize_species_name(raw_species)
        confidence = float(result.get("confidence", 0.0))
        model_version = result.get("model_version", "unknown")
        family = result.get("family") or SPECIES_TO_FAMILY.get(species, "unknown")

        conf_bucket = classify_confidence(confidence)
        apply_background(CONF_THEME[conf_bucket])

        # ---------------- RESULT SUMMARY ----------------
        st.success(
            f"""
            🌼 **Species:** {species.replace('_', ' ').title()}  
            🧬 **Family:** {family.replace('_', ' ').title()}  
            📊 **Confidence:** {confidence:.2f}  
            🧠 **Model Version:** {model_version}
            """
        )

        if conf_bucket == "bad":
            st.error("❌ **Poor match** (< 0.50).")
        elif conf_bucket == "medium":
            st.warning("⚠️ **Weak match** (0.50–0.69).")
        else:
            st.info("✅ **Strong match** (≥ 0.70).")

        # ---------------- IMAGES ----------------
        st.subheader("🖼️ Visual Reference")

        col1, col2 = st.columns(2)
        with col1:
            safe_image(FAMILY_IMAGE_MAP.get(family), f"{family.title()} family")
        with col2:
            safe_image(
                SPECIES_IMAGE_MAP.get(species),
                f"{species.replace('_', ' ').title()} species",
            )

        # ---------------- PROBABILITIES ----------------
        probs = result.get("probabilities")
        if isinstance(probs, dict) and probs:
            st.subheader("📊 Class probability breakdown")
            for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- **{k.replace('_', ' ').title()}**: {float(v):.2f}")

        # ---------------- INTERPRETATION ----------------
        st.subheader("🧠 Interpretation")
        st.markdown(
            """
            Confidence reflects how decisively the model can separate this input
            from other known classes. Lower confidence indicates overlapping
            feature ranges across species.
            """
        )

    except Exception as e:
        LOGGER.error(e)
        apply_background(CONF_THEME["offline"])
        st.error("Error communicating with backend. Check backend status and logs.")


if __name__ == "__main__":
    run()
