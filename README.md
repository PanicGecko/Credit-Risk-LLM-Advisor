# Credit Risk LLM Advisor

A machine learning app that predicts loan default risk through a chat interface. You describe a loan applicant in plain English, and the system parses your input, runs it through an XGBoost classifier trained on 1.35 million Lending Club loans, and responds with a risk assessment and explanation.

Built as a capstone project for the TripleTen ML Engineering bootcamp. It covers the full ML lifecycle: data preprocessing, experiment tracking, model selection, LLM-powered tool calling, testing, containerization, and deployment.

Intended for loan officers, credit analysts, or anyone evaluating peer-to-peer lending risk.

## Setup

### Prerequisites

- Python 3.11+
- An OpenAI API key from [platform.openai.com](https://platform.openai.com)
- Docker (optional but recommended)

### Option A: Docker (recommended)

```bash
git clone https://github.com/PanicGecko/Credit-Risk-LLM-Advisor.git
cd Credit-Risk-LLM-Advisor

cp .env.example .env
# open .env and paste your OpenAI API key

docker compose up --build
```

The app will be running at `http://localhost:8501`.

### Option B: Local

```bash
git clone https://github.com/PanicGecko/Credit-Risk-LLM-Advisor.git
cd Credit-Risk-LLM-Advisor

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# open .env and paste your OpenAI API key

streamlit run app.py
```

### Pulling data with DVC

The dataset and trained model are tracked with DVC and stored in a public S3 bucket. No AWS credentials needed.

```bash
dvc pull
```

For CI or quick testing, you can pull just the small test fixture instead:

```bash
dvc pull data/test_fixture.csv.dvc
```

## Usage

Open `http://localhost:8501` in your browser. The chat interface shows a welcome card with example prompts to get started.

The system needs three pieces of information to make a prediction: FICO score, loan amount, and annual income. If any of those are missing, it will ask for them. Additional features like DTI, employment length, loan purpose, and home ownership improve accuracy but are handled automatically when not provided.

### Example conversations

**Full prediction:**
> "FICO 710, requesting $15,000 for debt consolidation, earns $62K/year"

The advisor parses the features, runs the model, and responds with a default probability, risk classification (Low, Moderate, or Elevated), and a brief explanation.

**Missing information:**
> "What's the risk for someone with a 630 FICO?"

Loan amount and income are missing, so the advisor asks a follow-up question instead of guessing.

**Model questions:**
> "How accurate is the model?"

The advisor answers factual questions about the model's performance, training data, and methodology.

### Running tests

```bash
pytest tests/ -v
```

There are 43 tests across three files covering preprocessing, model predictions, and the LLM interface. Tests that need the trained model artifact are automatically skipped if it's not present.

## Architecture

The system has three layers: a Streamlit frontend, a Python backend that handles feature engineering and model inference, and the OpenAI API for natural language understanding.

```
Browser  -->  Streamlit server (Docker)  -->  OpenAI API
              - XGBoost pipeline in memory
              - Preprocessing + feature eng.
              - Tool-call dispatch
```

The LLM integration uses OpenAI's tool calling pattern. The GPT model receives a `predict_loan_default` function definition with the full feature schema. When the user provides enough information, GPT emits a structured tool call. The backend intercepts it, runs `predict_default_probability()`, and feeds the result back to GPT for a natural language explanation. Two API calls at most per prediction.

### Key files

| File | What it does |
|---|---|
| `app.py` | Streamlit frontend, chat UI, sidebar, custom CSS |
| `src/llm_interface.py` | OpenAI SDK orchestration, tool dispatch, conversation history |
| `src/model_predict.py` | Pipeline loading, feature engineering, prediction logic |
| `src/prompts.py` | System prompt and tool schema definition |
| `configs/config.yaml` | All training hyperparameters, no hardcoded values in code |
| `tests/` | pytest suite: 19 preprocessing, 11 model, 15 interface |

### Infrastructure

DVC handles data versioning with a dual-remote setup: a public-read S3 bucket for `dvc pull` and an authenticated remote for writes. MLflow logs all experiment runs to `mlruns/`. Docker uses a single-stage `python:3.11-slim` image with `libgomp1` for XGBoost, and the model artifact is baked in at build time.

## Results

### Model performance

The final model is XGBoost, selected after comparing four algorithms. All were trained with the same preprocessing pipeline and evaluated on a held-out 20% test set.

| Metric | XGBoost | Random Forest | HistGB | Logistic Reg. |
|---|---|---|---|---|
| AUC-ROC | 0.678 | 0.671 | 0.674 | 0.665 |
| F1 | 0.402 | 0.388 | 0.395 | 0.371 |
| Precision | 0.292 | 0.281 | 0.287 | 0.268 |
| Recall | 0.645 | 0.632 | 0.640 | 0.602 |

### Why the AUC tops out at 0.68

Three different model families converged within 0.01 AUC of each other. That means 0.68 is a data ceiling, not a model problem. The Sanz-Guerrero and Arroyo (2024) paper that curated this dataset reaches 0.71-0.74 by adding loan description text features, but those columns are over 90% missing in our data so they weren't worth the tradeoff.

When multiple algorithms plateau at the same number, the bottleneck is the feature set, not the model. More complexity wouldn't help here.

### Threshold tuning

The model uses `scale_pos_weight=1` with post-hoc threshold tuning at 0.199 rather than `class_weight='balanced'`, which degraded AUC by about 0.01 by distorting the probability ranking. Decoupling ranking quality from the operating point is standard practice in credit scoring. The threshold was selected to maximize F1 on the validation set.

The precision of 0.29 is intentionally low. The model is tuned as a screening tool: it catches 65% of actual defaults (recall) at the cost of more false alarms. In lending, missing a default is far more expensive than flagging a good loan for review. A loan officer uses this as one signal among many, not as a final decision.

### Preprocessing

Features are organized into three tiers (mandatory, recommended, optional) to mirror how a production API would handle partial data. Missing indicators are generated for all numeric features so the model can learn from missingness itself. Engineered ratios like loan-to-income and monthly payment burden capture relative financial stress, clipped at the 99th percentile to limit outlier influence.

## Reflection

### What I learned

The hardest decisions in this project were scope decisions, not technical ones. Recognizing the 0.68 AUC ceiling and stopping model iteration there, choosing tool calling over prompt chaining for the LLM interface, and structuring features into tiers that degrade gracefully with missing data. Those choices shaped the project more than any hyperparameter.

The tool calling pattern was a highlight. Instead of writing regex parsers or chained prompts, defining a function schema and letting the model decide when to call it produced something much cleaner. The LLM handles clarifying questions, partial input, and off-topic queries on its own, which would have required a lot of explicit control flow otherwise.

MLflow experiment tracking, even on a solo project, made the model selection narrative easy to write. Being able to programmatically compare runs and justify the final choice was worth the setup cost.

### What was challenging

Class imbalance was the most time-consuming iteration loop. I tried balanced class weights, scaled positive weights, and SMOTE before discovering that unweighted training with threshold tuning gave the best AUC. The intuition that class weighting distorts probability calibration and hurts ranking was not obvious upfront.

The DVC dual-remote configuration also took more debugging than expected. Getting a public-read S3 bucket that works with anonymous `dvc pull` while keeping authenticated writes on a separate remote required several rounds of IAM and bucket policy iteration.

### What I would improve with more time

SHAP integration would let the LLM cite which features drove each prediction up or down. A batch prediction mode accepting CSVs would be useful for portfolio analysis. On the infrastructure side, a GitHub Actions CI pipeline running tests on every push and a proper A/B testing framework for threshold values would make the system more production-ready. And for the ~10% of loans that do have description text, an NLP layer could push past the 0.68 ceiling.
