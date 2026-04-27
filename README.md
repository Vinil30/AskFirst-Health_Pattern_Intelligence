# Ask First - Health Pattern Detector
#Note: After opening the site either by local setup or directly using deployed site, press on run temporal reasoning to print all the outputs. Alongside, the improvements, you can toggle down the reasoning trace and detected patterns for each user. Please go through these things for better and efficient experience.
This project detects cross-conversation health patterns with temporal reasoning, assigns confidence scores, and structures output using Groq (OpenAI-compatible endpoint).

## 1) Setup

### Create and activate virtual environment
```powershell
python -m venv Environment
.\Environment\Scripts\Activate.ps1
```

### Install dependencies
```powershell
pip install -r requirements.txt
```

## 2) Configure `.env`

Create/update `.env` in project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

- `GROQ_API_KEY`: your Groq API key
- `GROQ_MODEL`: Groq model name to use

## 3) Run the app

```powershell
python app.py
```

It will open Streamlit automatically.

If not, run directly with Streamlit:

```powershell
streamlit run app.py
```

## 4) Hosted link (optional)

The app is already deployed, use this link:

`https://askfirst-health-pattern-intelligence.onrender.com/`

Make sure to run temporal reasoning to get all outputs.

