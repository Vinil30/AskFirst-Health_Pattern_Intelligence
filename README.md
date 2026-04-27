# Ask First - Health Pattern Detector
##Note: After opening the site either by local setup or directly using deployed site, press on run temporal reasoning to print all the outputs. Alongside, the improvements, you can toggle down the reasoning trace and detected patterns for each user. Please go through these things for better and efficient experience.
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

## 4) Hosted link

The app is already deployed, use this link:

`https://askfirst-health-pattern-intelligence.onrender.com/`

Make sure to run temporal reasoning to get all outputs.
Along with all the requirements provided in the assessment, I also added GenerativeAI part, for better reasoning and efficient suggestions, which provides a perfect touch to this system in terms of text and insight generation. I used llama70B model in this assessment, as it is trained on a good amount of health data, this LLM would be perfect for our work.

## 5) Folder structure

```text
Assignment_Internshala/
├── app.py
├── askfirst_synthetic_dataset.json
├── requirements.txt
├── .env
├── README.md
└── utils/
    ├── __init__.py
    ├── data_loader.py
    ├── feature_engineering.py
    ├── model_utils.py
    ├── groq_structurer.py
    ├── ModelTrainer.ipynb
    └── relation_model.pkl
```

File purpose summary:
- `app.py`: Streamlit UI + orchestration for model loading/training, scoring, and output rendering
- `askfirst_synthetic_dataset.json`: synthetic health conversation dataset
- `.env`: configuration for Groq credentials and model name
- `utils/data_loader.py`: dataset parsing and timeline construction
- `utils/feature_engineering.py`: candidate relation generation and temporal feature calculation
- `utils/model_utils.py`: training, saving, loading, and confidence scoring
- `utils/groq_structurer.py`: Groq/OpenAI endpoint integration and structured streaming JSON
- `utils/ModelTrainer.ipynb`: notebook to train and save model artifact
- `utils/relation_model.pkl`: saved model bundle used by the app

## 6) Workflow

1. Load dataset and select user scope (`ALL` or single user).
2. Build chronological timeline per user from session timestamps.
3. Create candidate tag relations and compute temporal features (lag, precedence, support, lift).
4. Train or load the saved model (`relation_model.pkl`).
5. Score patterns with confidence and generate one-line justification.
6. Show reasoning trace, detected patterns, and improvement suggestions in UI.
7. Send results to Groq endpoint for strict JSON structuring (with fallback if key missing).
8. Stream final structured output and allow JSON download.

## 7) Project explanation

This project is an end-to-end health reasoning system focused on cross-conversation intelligence.  
Instead of only keyword matching, it tries to detect patterns that depend on timing and repetition across sessions. The ML layer estimates relation confidence, while the LLM layer improves structure and readability of outputs in strict JSON format.

Core capabilities:
- Temporal pattern detection from multi-session user history
- Confidence scoring with explicit justification
- Reasoning trace visibility for transparency
- Suggestion generation for improvement and further analysis
- Streamlit interface for easy interaction and result export
