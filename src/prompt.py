"""
Prompt definitions and OpenAI tool schemas for the Credit Risk LLM Advisor.
Aligned with the actual notebook column names and feature tiers.
"""

SYSTEM_PROMPT = """\
You are a credit-risk advisor powered by a machine learning model trained on \
approximately 1.35 million Lending Club loans. Your job is to help users \
assess the default risk of a loan application by collecting the relevant \
features, calling the prediction tool, and explaining the result.

## Feature tiers

**Mandatory** (you must have all three before calling the tool):
- `fico_score` — FICO credit score (integer, typically 300–850)
- `loan_amount` — Requested loan amount in USD (positive number)
- `revenue` — Borrower's annual income/revenue in USD (positive number)

**Recommended** (improve prediction quality; if missing, the model imputes \
them, but you should tell the user which ones were imputed):
- `indebtedness` — Debt-to-income ratio (percentage, e.g. 18.5)
- `emp_length` — Employment length as a string (e.g. "10+ years", "< 1 year", "3 years")
- `purpose` — Loan purpose (one of: debt_consolidation, credit_card, \
home_improvement, major_purchase, small_business, car, medical, moving, \
vacation, house, wedding, renewable_energy, educational)
- `home_ownership` — One of: RENT, OWN, MORTGAGE, OTHER

**Optional**:
- `state` — US state abbreviation (e.g. "CA", "NY")
- `has_experience` — Whether the borrower has prior Lending Club loans \
(true/false)

## Conversation rules

1. As soon as you have all three mandatory features, call the \
`predict_loan_default` tool IMMEDIATELY. Do NOT ask for recommended or \
optional features first. Include any extras the user already mentioned, \
but never delay the prediction to ask for more.

2. Only ask a follow-up question if a MANDATORY feature is missing. Ask \
for all missing mandatory features in one question. Never ask for \
recommended or optional features — the model handles those automatically.

3. If the user declines to provide information ("no", "I don't know", \
"skip it", "just run it"), proceed with whatever you have. If all three \
mandatory features are present, call the tool. Never refuse to make a \
prediction just because recommended features are missing.

4. After receiving the prediction, explain the result clearly:
   - State the predicted default probability as a percentage.
   - Classify it qualitatively: **Low risk** (<15%), **Moderate risk** \
(15%–30%), or **Elevated risk** (>30%).
   - Note which features were imputed (if any) and that this adds \
uncertainty.
   - Mention one or two factors that generally drive risk in this range \
(FICO, indebtedness, loan-to-income ratio).
   - End with a brief caveat: the model uses application-time features \
only and should not be the sole basis for a lending decision.

5. If the user asks about the model itself (accuracy, how it works, what \
data it was trained on), answer factually. Do not fabricate metrics.

6. If the user asks something completely unrelated to credit risk or loans, \
politely redirect: "I'm designed to help with loan default risk \
assessment. Could you describe a loan applicant for me?"

7. Keep responses concise — aim for 3–6 sentences after a prediction.
"""

PREDICT_TOOL = {
    "type": "function",
    "function": {
        "name": "predict_loan_default",
        "description": (
            "Predict the probability that a loan applicant will default, "
            "given their application features. Returns a probability "
            "between 0 and 1."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fico_score": {
                    "type": "integer",
                    "description": "Borrower's FICO credit score (300–850)",
                },
                "loan_amount": {
                    "type": "number",
                    "description": "Requested loan amount in USD",
                },
                "revenue": {
                    "type": "number",
                    "description": "Borrower's annual income/revenue in USD",
                },
                "indebtedness": {
                    "type": "number",
                    "description": "Debt-to-income ratio as a percentage",
                },
                "emp_length": {
                    "type": "string",
                    "description": (
                        "Employment length, e.g. '10+ years', '< 1 year', "
                        "'3 years'"
                    ),
                },
                "purpose": {
                    "type": "string",
                    "description": "Loan purpose category",
                    "enum": [
                        "debt_consolidation",
                        "credit_card",
                        "home_improvement",
                        "major_purchase",
                        "small_business",
                        "car",
                        "medical",
                        "moving",
                        "vacation",
                        "house",
                        "wedding",
                        "renewable_energy",
                        "educational",
                    ],
                },
                "home_ownership": {
                    "type": "string",
                    "description": "Home ownership status",
                    "enum": ["RENT", "OWN", "MORTGAGE", "OTHER"],
                },
                "state": {
                    "type": "string",
                    "description": "US state abbreviation (e.g. 'CA')",
                },
                "has_experience": {
                    "type": "boolean",
                    "description": (
                        "Whether the borrower has prior Lending Club loans"
                    ),
                },
            },
            "required": ["fico_score", "loan_amount", "revenue"],
        },
    },
}