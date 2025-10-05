# Deploy to Render

This repo is ready for one-click deployment to Render as a Web Service. It uses Gunicorn to serve the Flask app and avoids long cold starts by allowing you to ship pre-trained models.

## Prereqs
- A GitHub (or GitLab) repo containing this project
- Python 3.13 compatible environment (Render native Python runtime is fine)

## Recommended: train locally and commit models
1) Create and activate your venv, install deps, train and save models:
   - Windows PowerShell
     - `& .\\venv\\Scripts\\Activate.ps1`
     - `pip install -r requirements.txt`
     - `python .\\tune_and_train.py`
2) Ensure the following files exist in `models/`:
   - `exoplanet_ensemble_rf.pkl`
   - `exoplanet_ensemble_xgb.pkl`
   - `exoplanet_ensemble_lgb.pkl`
   - `exoplanet_ensemble_cnn.h5`
   - `exoplanet_ensemble_metadata.pkl`
   - `exoplanet_ensemble_scaler.pkl`
   - `model_metrics.json`
3) Commit and push these to your repo. Note: If model files are very large, consider Git LFS or using Render Disks instead of committing binaries.

## Create the Render service (native Python runtime)
1) In Render, click New → Web Service → Connect your repo
2) Use these settings:
   - Runtime: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --workers=2 --threads=4 --timeout=120 --preload`
3) Environment variables:
   - `PYTHONUNBUFFERED=1`
   - `PORT=5000`
   - `SKIP_TRAINING=1` (prevents training on cold start; remove/set to 0 only if you want to train on the server)
   - Optionally `FLASK_DEBUG=0`
4) Click Create Web Service and wait for it to build and boot.

If you’re using `render.yaml`, Render can auto-detect and prefill most of this.

## Persist models and data (recommended for on-host training)
- This repo includes a `render.yaml` that provisions a Render Disk mounted at `/var/data`.
- The app reads/writes using env vars:
   - `DATA_DIR` (defaults to `.`) → where `cumulative_data*.csv` are located
   - `MODELS_DIR` (defaults to `models/`) → where trained models and metrics are saved
   - `UPLOADS_DIR` (defaults to `uploads/`) → where user-uploaded CSVs go
- On Render, `render.yaml` sets these to `/var/data`, `/var/data/models`, and `/var/data/uploads` respectively, so they persist across deploys.

## Verifying
- After deploy, open the Render URL. The app should load models from `models/` and be immediately responsive.
- API endpoints:
  - `/` home
  - `/api/metrics` to view current metrics
   - `/api/train/start` to kick off background training (requires `ADMIN_TOKEN`)
   - `/api/train/status` to check background training status
   - `/api/retrain` to run synchronous training (blocks request; not recommended on small instances; requires `ADMIN_TOKEN`)

### Trigger background training on Render
1) In the Render service, copy the value of the auto-generated `ADMIN_TOKEN` env var (or set your own token).
2) Start training (replace SERVICE_URL and TOKEN):
    - Windows PowerShell example:
       - `Invoke-RestMethod -Method Post -Uri "$env:SERVICE_URL/api/train/start?token=$env:TOKEN"`
3) Check status:
    - `Invoke-RestMethod -Method Get -Uri "$env:SERVICE_URL/api/train/status"`

## Troubleshooting
- Memory/timeouts: reduce Gunicorn workers to 1
  - Start Command: `gunicorn app:app --workers=1 --threads=4 --timeout=180 --preload`
- Large models: use Disks or Git LFS. Some hosts don’t fetch LFS by default in build—consider Disks or downloading artifacts at runtime.
- Training on host: set `SKIP_TRAINING=0` (can be slow and resource-intensive). Consider a one-off job to train and persist to a Disk.
   - Alternatively, leave `SKIP_TRAINING=1`, then use `/api/train/start` to run training in the background while web stays responsive.

## Containerized alternative
If preferred, you can deploy using the included `Dockerfile`. Choose “Use Docker” when creating the Render service.
- The Docker image exposes port 5000 and starts Gunicorn.
- `libgomp1` is installed for LightGBM/XGBoost.

---
This project is configured to read `PORT`, `FLASK_DEBUG` and `SKIP_TRAINING` at runtime and will load a persisted scaler (`models/exoplanet_ensemble_scaler.pkl`) if present to avoid heavy preprocessing on startup.
