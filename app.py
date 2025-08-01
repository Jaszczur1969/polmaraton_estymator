import os
import io
import pickle
import streamlit as st
import pandas as pd
import boto3
from dotenv import load_dotenv
from openai import OpenAI
import requests
from datetime import timedelta

# --- Load environment ---
load_dotenv()

# OpenAI client
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Langfuse config
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")  # e.g., https://api.langfuse.com/v1/track
LANGFUSE_PROJECT = os.getenv("LANGFUSE_PROJECT", "polmaraton_estymator")

# DigitalOcean Spaces / S3 client
SESSION = boto3.session.Session()
s3 = SESSION.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3"),
    )

BUCKET_NAME="wiadro-jaszczur1969"
MODEL_PATH ="model/Linear_Regression_pipeline.pkl"


# Cache the model loading
@st.cache_resource(show_spinner=False)
def load_model_from_spaces(bucket: str, key: str):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        model = pickle.load(io.BytesIO(data))
        return model
    except Exception as e:
        st.error(f"Nie udało się załadować modelu z Spaces: {e}")
        return None

# Utility: convert seconds to H:MM:SS
def format_seconds_to_hms(sec_float):
    sec = int(round(sec_float))
    return str(timedelta(seconds=sec))

# Langfuse logging (basic)
def log_to_langfuse(event_name: str, payload: dict):
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_HOST:
        return  # nie logujemy jeśli brak konfiguracji
    headers = {
        "Authorization": f"Bearer {LANGFUSE_PUBLIC_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "project": LANGFUSE_PROJECT,
        "event": event_name,
        "payload": payload,
        "timestamp": None,  # pozwól backendowi ustawić
    }
    try:
        resp = requests.post(LANGFUSE_HOST, json=body, headers=headers, timeout=5)
        if resp.status_code >= 400:
            st.warning(f"Langfuse logging failed: {resp.status_code} {resp.text}")
    except Exception:
        # nie przerywamy działania aplikacji z powodu Langfuse
        pass

# LLM extraction prompt
def extract_structured_data(free_text: str):
    system_prompt = (
        "Jesteś parsującym asystentem. Ze swobodnego opisu użytkownika wyciągnij dokładnie trzy pola:\n"
        "1. płeć_encoded: 0 jeśli kobieta, 1 jeśli mężczyzna (jeśli ujmuje słownie: kobieta, mężczyzna, pani, pan etc.)\n"
        "2. wiek: liczba całkowita\n"
        "3. 5_km_czas_sec: czas przebiegnięcia 5 km, skonwertowany do sekund. Użytkownik może podawać w formacie mm:ss, m:ss, np. '22:30', '23 minuty 10 sekund', '25 minut', '1300 sekund', '23.5 minut'.\n"
        "Odpowiedz tylko w czystym JSON-ie z kluczami: płeć_encoded, wiek, 5_km_czas_sec. Jeśli czegoś brakuje, nie zgaduj, daj wartość null dla brakującego pola."
    )
    user_prompt = free_text.strip()

    try:
        response = openai.chat.completions.create(
            model="gpt-4",  # możesz zmienić jeśli preferujesz inny
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()
        # Log prompt + output
        log_to_langfuse("llm_extraction", {
            "input_text": free_text,
            "system_prompt": system_prompt,
            "parsed_output_raw": content,
        })
        # Próba parsowania JSON
        try:
            parsed = pd.io.json.loads(content)
        except Exception:
            # fallback: próbujemy bezpośrednio eval (ostrożnie)
            try:
                parsed = eval(content, {"__builtins__": {}})
            except Exception:
                parsed = {}
        return parsed, content
    except Exception as e:
        st.error(f"Błąd podczas wywołania OpenAI: {e}")
        log_to_langfuse("llm_extraction_error", {"error": str(e), "input_text": free_text})
        return {}, ""

# Sprawdzanie brakujących pól
def missing_fields(parsed: dict):
    required = ["płeć_encoded", "wiek", "5_km_czas_sec"]
    missing = []
    for k in required:
        if k not in parsed or parsed[k] in (None, "", []):
            missing.append(k)
    return missing

# Główna aplikacja Streamlit
def main():
    st.set_page_config(page_title="Estymator półmaratonu", page_icon="🏃‍♂️")
    st.title("Estymator czasu ukończenia półmaratonu")

    st.markdown(
        "Wprowadź w polu tekstowym swój opis, np.: "
        "_Cześć, mam 29 lat, jestem mężczyzną, na 5 km biegam w 23:10._"
    )

    with st.form("input_form"):
        free_text = st.text_area("Opisz siebie (płeć, wiek, czas 5 km)", height=150)
        submitted = st.form_submit_button("Estymuj czas")
        clear = st.form_submit_button("Czyść dane")

    if clear:
        st.experimental_rerun()

    if submitted:
        if not free_text.strip():
            st.warning("Podaj opis zawierający płeć, wiek i czas na 5 km.")
            return

        with st.spinner("Analizuję opis..."):
            parsed, raw_llm_output = extract_structured_data(free_text)

        # Uproszczona normalizacja: próbujemy rzutować
        normalized = {}
        # płeć
        try:
            sex = parsed.get("płeć_encoded", None)
            if sex is not None:
                normalized["płeć_encoded"] = int(sex)
        except:
            normalized["płeć_encoded"] = None
        # wiek
        try:
            age = parsed.get("wiek", None)
            if age is not None:
                normalized["wiek"] = int(age)
        except:
            normalized["wiek"] = None
        # 5 km czas
        try:
            t5 = parsed.get("5_km_czas_sec", None)
            if t5 is not None:
                normalized["5_km_czas_sec"] = float(t5)
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
            log_to_langfuse("incomplete_input", {
                "normalized": normalized,
                "missing": miss
            })
            return

        # Przygotuj df do modelu
        input_df = pd.DataFrame([{
            "płeć_encoded": normalized["płeć_encoded"],
            "Wiek": normalized["wiek"],
            "5_km_czas_sec": normalized["5_km_czas_sec"],
        }])

        st.subheader("Dane wejściowe dla modelu")
        st.table(input_df)

        # Załaduj model
        model = load_model_from_spaces(BUCKET_NAME, MODEL_PATH)
        if model is None:
            st.error("Model nie został załadowany.")
            return

        # Estymacja
        try:
            # W zależności od tego, czy model był trenowany przez pycaret.regression lub inny
            # Zakładamy, że model przyjmuje DataFrame i ma .predict
            est_sec = model.predict(input_df)[0]
            formatted = format_seconds_to_hms(est_sec)

            st.success("Estymacja ukończenia półmaratonu:")
            st.metric("Szacowany czas (H:MM:SS)", formatted)
            st.write(f"(czyli około **{est_sec:.1f} sekund**)")

            # Log sukcesu
            log_to_langfuse("prediction_success", {
                "input": input_df.to_dict(orient="records")[0],
                "predicted_seconds": est_sec,
                "formatted": formatted,
                "llm_parsed_raw": raw_llm_output,
            })
        except Exception as e:
            st.error(f"Błąd podczas predykcji: {e}")
            log_to_langfuse("prediction_error", {"error": str(e), "input": input_df.to_dict(orient="records")[0]})

    st.markdown("---")
    st.caption("Aplikacja używa OpenAI do parsowania tekstu, modelu z PyCaret do estymacji oraz Langfuse do zbierania metryk.")

if __name__ == "__main__":
    main()
