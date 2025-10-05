# NASA Exoplanet Hunt — Streamlit UI

This repo contains both a Flask app (`app.py`) and a Streamlit front-end (`streamlit_app.py`). If you want the interactive sidebar and widgets to trigger updates, you must run the Streamlit app.

## Run locally (Windows PowerShell)

1. Install dependencies (already done if you ran `pip install -r requirements.txt`).
2. Start Streamlit:

```powershell
$env:DATA_DIR='.'; $env:MODELS_DIR='models'; $env:ALLOW_TRAINING='0'; streamlit run .\streamlit_app.py --server.port 8501
```

- Set `ALLOW_TRAINING='1'` if you want to enable retraining from the UI (can be slow).
- Optionally set `QUICK_TRAIN='1'` to speed up training for demos.

Then open the URL printed by Streamlit, e.g. http://localhost:8501

## VS Code task

Use the built-in task:

- Open Command Palette → "Run Task" → "Run Streamlit app"

This executes:

```
streamlit run ${workspaceFolder}/streamlit_app.py --server.port 8501 --server.headless true
```

## Notes

- Running `python app.py` starts the Flask site, not Streamlit. Streamlit widgets will not appear or rerun there.
- Terminal logs have been added to `streamlit_app.py` so interactions (form submit, retrain) print messages. Look for lines prefixed with `[INFO]`.
