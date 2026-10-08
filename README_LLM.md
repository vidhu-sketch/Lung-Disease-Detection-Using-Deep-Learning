LLM Integration — Quick Guide

1) Local setup
- Copy `.env.example` to `.env` and edit it with your OpenAI key (do NOT commit `.env`).
- From the project inner folder (where `app.py` lives):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

- Run the app locally:

```powershell
$env:OPENAI_API_KEY = "sk-..."  # or use .env
python app.py
```

2) Docker
- Do not bake secrets into images. Pass the key at runtime:

```powershell
docker build -t lung-cancer-app:latest .
docker run -e OPENAI_API_KEY="sk-..." -p 5000:5000 lung-cancer-app:latest
```

3) Heroku
- Set config var in Heroku dashboard or via CLI:

```powershell
heroku config:set OPENAI_API_KEY="sk-..."
```

4) Test the LLM call
- Quick test from project folder (will use `.env` or environment variable):

```powershell
python -c "from app import generate_llm_report; print(generate_llm_report('Normal', 98.5))"
```

5) API behavior
- `/api/predict` returns `llm_report` and `llm_error` fields. `llm_error` is `null` on success or one of:
  - `no_api_key`, `timeout`, `quota_exceeded`, `service_error`.

6) Security
- Keep `.env` out of source control. Use platform secrets for deployments.
