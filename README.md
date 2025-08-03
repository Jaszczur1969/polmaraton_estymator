# 🏃‍♂️ Estymator czasu w półmaratonie

**Opis:** Aplikacja Streamlit estymuje czas ukończenia półmaratonu na podstawie naturalnego opisu użytkownika (płeć, wiek, czas na 5 km). Parsowanie tekstu realizuje LLM (OpenAI) z automatycznym śledzeniem w Langfuse, a predykcja odbywa się z użyciem modelu PyCaret pobranego z DigitalOcean Spaces.

## 📦 Wymagania

- Python 3.8+
- Plik `requirements.txt` z zależnościami
- Klucze API do OpenAI i Langfuse
- Plik `.pkl` z wytrenowanym modelem PyCaret, załadowany na DigitalOcean Spaces (S3-compatible)

## ⚙️ Konfiguracja środowiska

Utwórz plik `.env` z zawartością:

```env
OPENAI_API_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## ☁️ Deployment na Digital Ocean App Platform

1. Stwórz nowe App w Digital Ocean App Platform.
2. Wskaż repozytorium z aplikacją.
3. Dodaj zmienne środowiskowe z `.env`.
4. Wybierz polecenie startowe:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## 🖥 Deployment na własnym serwerze (Droplet)

1. Utwórz Droplet z Pythonem 3.9+.
2. Sklonuj repozytorium i zainstaluj wymagania.
3. Uruchom:

```bash
nohup streamlit run app.py --server.port 8080 --server.address 0.0.0.0 &
```

4. Skonfiguruj reverse proxy (np. Nginx) dla publicznego dostępu.

## 🛠 Troubleshooting

- **Brak **`` – sprawdź, czy plik nie nazywa się `streamlit.py`, usuń `__pycache__`, zaktualizuj Streamlit.
- **Model nie ładuje się** – zweryfikuj `BUCKET_NAME`, `MODEL_PATH` i klucze S3.
- **Błędy OpenAI/Langfuse** – sprawdź klucze i czy `LANGFUSE_PROJECT` istnieje w panelu.

## 📩 Kontakt

W przypadku problemów lub chęci rozwoju aplikacji, skontaktuj się z autorem poprzez GitHub Issues lub e-mail.

