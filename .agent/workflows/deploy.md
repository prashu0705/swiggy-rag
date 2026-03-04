---
description: how to deploy the Swiggy RAG application to Streamlit Community Cloud
---

# Deploying to Streamlit Community Cloud

This guide explains how to deploy your RAG application for free using Streamlit's official hosting service.

## 1. Prepare Your Repository
Ensure your GitHub repository has the following structure:
- `app.py` (Main entry point)
- `ingest.py` (Ingestion logic)
- `requirements.txt` (Dependencies)
- `Annual-Report-FY-2023-24 (1) (1).pdf` (The source document)
- `.streamlit/config.toml` (Optional: for theme/server settings)

## 2. Push to GitHub
If you haven't already, push your code to a public or private GitHub repository.
```bash
git init
git add .
git commit -m "Prepare for deployment"
git branch -M main
git remote add origin <your-github-url>
git push -u origin main
```

## 3. Connect to Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Click **"New app"**.
3. Select your repository, branch (`main`), and main file path (`app.py`).

## 4. Configure Secrets (CRITICAL)
Your `.env` file is ignored or should be. You must add your API keys to the Streamlit Secrets manager:
1. In the app deployment page, click **"Settings"** -> **"Secrets"**.
2. Paste the following:
```toml
GOOGLE_API_KEY = "your_google_api_key_here"
GROQ_API_KEY = "your_groq_api_key_here"
OPENAI_API_KEY = "your_openai_api_key_here"
```

## 5. Handling the Vector Database
The current setup processes the PDF on the first run and stores it in `db_free/`. 
- **Ephemeral Storage**: Streamlit's disk is ephemeral. The vector DB will be lost if the app restarts.
- **Auto-Ingestion**: The script handles this by checking if `db_free` exists. It will re-ingest automatically on the first user request after a reboot.

## 6. Optimization for Deployment
- **Resources**: The app uses `HuggingFaceEmbeddings` which requires around 1GB of RAM for the model. Streamlit's free tier provides ~1GB, so it should fit, but if you hit memory limits, consider using a cloud embedding provider (like Google or OpenAI).
