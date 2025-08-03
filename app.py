import os
import streamlit as st
import pandas as pd
import boto3
from dotenv import load_dotenv
import requests  # nadal używane do ewentualnych fallbacków
from datetime import timedelta
from pycaret.regression import load_model as pycaret_load_model
import tempfile
from langfuse.openai import openai  # wrapper OpenAI (automatyczne LLM trace'y)
from langfuse.decorators import observe  # dekorator do spanów


st.set_page_config(page_title="Estymator półmaratonu", page_icon="🏃‍♂️") 
st.markdown(
    """
    <style>
      .custom-title { text-align: center; margin-bottom: 0.25rem; }
    </style>
    <h1 class="custom-title">Estymator czasu w półmaratonie 🏃‍♂️</h1>
    """,
    unsafe_allow_html=True,
)

# --- Ładujemy dane z pliku .env ---
load_dotenv()

# --- OpenAI + Langfuse wrapper setup (automatyczne śledzenie LLM) ---
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
openai.langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
openai.langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
openai.langfuse_enabled = True

# --- Inicjowanie dostępu do Digital Ocean/S3 client ---
SESSION = boto3.session.Session()
s3 = SESSION.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3"),
)
BUCKET_NAME = os.getenv("BUCKET_NAME", "wiadro-jaszczur1969")
MODEL_PATH = os.getenv("MODEL_PATH", "model/Linear_Regression_pipeline.pkl")

# --- Funkcja załadowania modelu ze Spaces (+ decorator) ---
@observe(name="model_load_from_spaces")
@st.cache_resource(show_spinner=False)
def load_model_from_spaces(bucket: str, key_with_ext: str):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key_with_ext)
        data = obj["Body"].read()
        with tempfile.TemporaryDirectory() as td:
            pkl_path = os.path.join(td, os.path.basename(key_with_ext))
            with open(pkl_path, "wb") as f:
                f.write(data)
            base_no_ext = os.path.splitext(pkl_path)[0]
            model = pycaret_load_model(base_no_ext)
        return model
    except Exception as e:
        st.error(f"Nie udało się załadować modelu z Spaces: {e}")
        # wyjątek zostanie automatycznie zapisany przez dekorator
        return None

# --- Funkcja do formatowania sekund na H:MM:SS ---
def format_seconds_to_hms(sec_float):
    sec = int(round(sec_float))
    return str(timedelta(seconds=sec))

# --- Funkcja do wyciągnięcie danych z opisu użytkownika z pomocą LLM (+ decorator) ---
@observe(name="llm_extract_structured_data")
def extract_structured_data(free_text: str):
    system_prompt = (
        "Jesteś parsującym asystentem. Ze swobodnego opisu użytkownika wyciągnij dokładnie trzy pola:\n"
        "1. płeć_encoded: 0 jeśli kobieta, 1 jeśli mężczyzna (jeśli ujmuje słownie: kobieta, mężczyzna, pani, pan etc.)\n"
        "2. wiek: liczba całkowita\n"
        "3. 5_km_czas_sec: czas przebiegnięcia 5 km, skonwertowany do sekund. Użytkownik może podawać w formacie mm:ss, m:ss, np. '22:30', '23 minuty 10 sekund', '25 minut', '1300 sekund', '23.5 minut'.\n"
        "Odpowiedz tylko w czystym JSON-ie z kluczami: płeć_encoded, wiek, 5_km_czas_sec. Jeśli czegoś brakuje, nie zgaduj, daj wartość null dla brakującego pola. Jeśli użytkownik poda rok urodzenia oblicz jego wiek, biorąc pod uwagę bieżący rok (teraz mamy 2025).\n"
    )
    user_prompt = free_text.strip()

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            name="llm_parse_input",
            metadata={"stage": "parse_input"},
        )
        content = response.choices[0].message.content.strip()
        # Próba parsowania JSON
        try:
            parsed = pd.io.json.loads(content)
        except Exception:
            try:
                parsed = eval(content, {"__builtins__": {}})
            except Exception:
                parsed = {}
        return parsed, content
    except Exception as e:
        st.error(f"Błąd podczas wywołania OpenAI: {e}")
        return {}, ""

# --- Funkcja do sprawdzania brakujących pól ---
def missing_fields(parsed: dict):
    required = ["płeć_encoded", "wiek", "5_km_czas_sec"]
    missing = []
    for k in required:
        if k not in parsed or parsed[k] in (None, "", []):
            missing.append(k)
    return missing


@observe(name="handle_incomplete_input")
def handle_incomplete(normalized: dict, miss: list):
    # tylko dla trace'u; nic dodatkowego nie musi robić
    return {"missing": miss, "normalized": normalized}


@observe(name="do_prediction")
def do_prediction(input_df, model, raw_llm_output):
    est_sec = model.predict(input_df)[0]
    formatted = format_seconds_to_hms(est_sec)
    return est_sec, formatted

# --- Główna aplikacja Streamlit ---
def main():



    st.markdown(
        "📌 **Wprowadź w polu tekstowym swój opis, np.: "
        "_Cześć, mam 29 lat, jestem mężczyzną, na 5 km biegam w 23:10._"
    )


    with st.form("input_form"):
        free_text = st.text_area("Opisz siebie (płeć, wiek, czas 5 km)", height=150)
        submitted = st.form_submit_button("Estymuj czas ⏱️")
        
    # --- Przycisk do czyszczenia sesji (danych) ---
    clear = st.button("Czyść dane") 
    if clear:
        for key in ["free_text", "parsed", "normalized", "model_loaded"]:
            if key in st.session_state:
                del st.session_state[key]
        return

    if submitted:
        if not free_text.strip():
            st.warning("Podaj opis zawierający płeć, wiek i czas na 5 km.")
            return

        with st.spinner("Analizuję opis..."):
            parsed, raw_llm_output = extract_structured_data(free_text)

        # --- Uproszczona normalizacja: próbujemy rzutować ---
        normalized = {}

        # płeć
        try:
            sex = parsed.get("płeć_encoded", None)
            normalized["płeć_encoded"] = int(sex) if sex is not None else None
        except:
            normalized["płeć_encoded"] = None

        # wiek
        try:
            age = parsed.get("wiek", None)
            normalized["wiek"] = int(age) if age is not None else None
        except:
            normalized["wiek"] = None

        # 5 km czas
        try:
            t5 = parsed.get("5_km_czas_sec", None)
            normalized["5_km_czas_sec"] = float(t5) if t5 is not None else None
        except:
            normalized["5_km_czas_sec"] = None

        miss = missing_fields(normalized)
        if miss:
            st.error(f"Brakuje danych potrzebnych do estymacji: {', '.join(miss)}")
            st.json({
                "parsowane_raw": parsed,
                "znormalizowane": normalized,
                "brakujące_pola": miss,
                "surowa_odpowiedź_LLM": raw_llm_output,
            })
            handle_incomplete(normalized, miss)
            return
        
        # --- Przygotuj df do modelu (surowe wartości dla predykcji) ---
        input_df = pd.DataFrame([{
            "płeć_encoded": normalized["płeć_encoded"],
            "Wiek": normalized["wiek"],
            "5_km_czas_sec": normalized["5_km_czas_sec"],
        }])

        # --- Przygotuj wersję do wyświetlenia użytkownikowi ---
        def decode_sex(code):
            if code == 1:
                return "Mężczyzna"
            if code == 0:
                return "Kobieta"
            return None 
        
        display_df = pd.DataFrame([{
            "Płeć": decode_sex(normalized["płeć_encoded"]),
            "Wiek": normalized["wiek"],  
            "Czas na 5 km [H:MM:SS]": format_seconds_to_hms(normalized["5_km_czas_sec"]) if normalized["5_km_czas_sec"] is not None else None,
        }])


        st.subheader("Dane wejściowe dla modelu")
        st.table(display_df)

        # --- Załaduj model ---
        model = load_model_from_spaces(BUCKET_NAME, MODEL_PATH)
        if model is None:
            st.error("Model nie został załadowany.")
            return
        
        # --- Estymacja ---
        try:
            est_sec, formatted = do_prediction(input_df, model, raw_llm_output)
            st.success("Estymacja ukończenia półmaratonu:")
            st.metric("Szacowany czas (H:MM:SS)", formatted)
            st.write(f"(czyli około **{est_sec:.1f} sekund**)")

        except Exception as e:
            st.error(f"Błąd podczas predykcji: {e}")
            # wyjątek automatycznie zarejestrowany przez dekorator

    # --- Krótkie info o sposobie działania aplikacji ---
    st.markdown("---")
    st.caption(
        "Aplikacja używa Langfuse-wrapped OpenAI do parsowania tekstu oraz dekoratorów `@observe` z Langfuse SDK v3 do dodatkowych metryk."
    )


if __name__ == "__main__":
    main()