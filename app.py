"""
Credit Risk LLM Advisor — Streamlit Frontend
Editorial minimalist chat interface for loan default prediction.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from llm_interface import get_response

# ─── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Advisor",
    page_icon="◆",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Design tokens ──────────────────────────────────────────────────
# Fraunces (serif display) + IBM Plex Sans (body)
# Deep forest green on warm off-white
CUSTOM_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,400&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&display=swap');

  /* ── Root variables ─────────────────────────────────── */
  :root {
    --green-900:  #1B3A2D;
    --green-800:  #234D3B;
    --green-700:  #2C614A;
    --green-600:  #3A7D5E;
    --green-500:  #4A9A73;
    --green-100:  #D4E8DC;
    --green-50:   #EBF4EF;
    --cream:      #FAF8F4;
    --cream-dark: #F0EDE6;
    --warm-gray:  #8A8578;
    --text:       #2B2924;
    --text-muted: #6B675E;
    --text-light: #9E9A90;
    --red-soft:   #C4503A;
    --amber-soft: #B8860B;
    --serif:      'Fraunces', 'Georgia', serif;
    --sans:       'IBM Plex Sans', 'Helvetica Neue', sans-serif;
    --radius:     10px;
    --shadow-sm:  0 1px 3px rgba(27,58,45,0.06);
    --shadow-md:  0 4px 16px rgba(27,58,45,0.08);
    --transition: 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  }

  /* ── Global resets ──────────────────────────────────── */
  .stApp,
  .stApp > div,
  .stApp [data-testid="stAppViewContainer"],
  .stApp [data-testid="stAppViewBlockContainer"],
  .stApp main,
  .stApp section[data-testid="stMain"],
  .stApp [data-testid="stVerticalBlock"],
  .stApp [data-testid="stChatMessageContent"] {
    background-color: var(--cream) !important;
    color: var(--text) !important;
  }
  .stApp {
    font-family: var(--sans) !important;
  }

  /* Hide ALL default Streamlit chrome — header, footer, deploy, hamburger */
  header[data-testid="stHeader"] {
    background: transparent !important;
    backdrop-filter: none !important;
  }
  #MainMenu,
  footer,
  .stDeployButton,
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stStatusWidget"],
  .viewerBadge_container__r5tak,
  .stActionButton,
  ._profileContainer_gzau3_53,
  [data-testid="manage-app-button"],
  [data-testid="stAppDeployButton"],
  header .stActionButton {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    position: absolute !important;
    pointer-events: none !important;
  }

  /* ── Typography ─────────────────────────────────────── */
  h1, h2, h3 {
    font-family: var(--serif) !important;
    color: var(--green-900) !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
  }
  h1 { font-size: 2.1rem !important; line-height: 1.15 !important; }

  p, li, span, div, input, textarea, label, .stMarkdown {
    font-family: var(--sans) !important;
  }

  /* ── Chat input ─────────────────────────────────────── */
  /* Kill the bottom bar background that Streamlit adds */
  [data-testid="stBottom"],
  [data-testid="stBottom"] > div {
    background: var(--cream) !important;
    border-top: 1px solid var(--cream-dark) !important;
  }

  /* The entire input wrapper */
  .stChatInput {
    background: transparent !important;
    border: none !important;
  }

  /* Outer container — the visible rounded box */
  .stChatInput > div {
    background: #FFFFFF !important;
    border: 1.5px solid var(--cream-dark) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: border-color var(--transition), box-shadow var(--transition) !important;
  }
  .stChatInput > div:focus-within {
    border-color: var(--green-600) !important;
    box-shadow: 0 0 0 3px rgba(58,125,94,0.12) !important;
  }

  /* Force EVERY nested element inside the input to be white/transparent */
  .stChatInput div,
  .stChatInput form,
  .stChatInput [data-testid] {
    background: transparent !important;
    background-color: transparent !important;
  }

  /* The textarea itself */
  .stChatInput textarea {
    background: transparent !important;
    background-color: transparent !important;
    font-family: var(--sans) !important;
    font-size: 0.95rem !important;
    color: var(--text) !important;
    caret-color: var(--green-700) !important;
    -webkit-text-fill-color: var(--text) !important;
  }
  .stChatInput textarea::placeholder {
    color: var(--text-light) !important;
    -webkit-text-fill-color: var(--text-light) !important;
    font-style: italic;
    opacity: 1 !important;
  }

  /* Send button inside the input */
  .stChatInput button,
  .stChatInput [data-testid="stChatInputSubmitButton"] {
    background: var(--green-900) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    transition: background var(--transition) !important;
  }
  .stChatInput button:hover,
  .stChatInput [data-testid="stChatInputSubmitButton"]:hover {
    background: var(--green-700) !important;
  }
  .stChatInput button svg {
    fill: white !important;
    stroke: white !important;
  }

  /* ── Chat message bubbles ───────────────────────────── */
  [data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 0 !important;
    max-width: 780px;
    margin: 0 auto;
  }

  /* User messages */
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
  }
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {
    background: var(--green-900) !important;
    color: var(--green-50) !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 0.85rem 1.15rem !important;
    box-shadow: var(--shadow-sm);
    max-width: 80%;
    margin-left: auto;
    font-size: 0.925rem;
    line-height: 1.6;
  }
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] p {
    color: var(--green-50) !important;
  }

  /* Assistant messages */
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] {
    background: #FFFFFF !important;
    border: 1px solid var(--cream-dark) !important;
    border-radius: 16px 16px 16px 4px !important;
    padding: 0.85rem 1.15rem !important;
    box-shadow: var(--shadow-sm);
    max-width: 85%;
    font-size: 0.925rem;
    line-height: 1.65;
    color: var(--text) !important;
  }
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] p,
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] li,
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] span {
    color: var(--text) !important;
  }
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] strong {
    color: var(--green-800) !important;
    font-weight: 600;
  }

  /* Avatar styling */
  [data-testid="chatAvatarIcon-assistant"] {
    background: var(--green-700) !important;
    color: white !important;
  }
  [data-testid="chatAvatarIcon-user"] {
    background: var(--cream-dark) !important;
    color: var(--text-muted) !important;
  }

  /* ── Sidebar ────────────────────────────────────────── */
  section[data-testid="stSidebar"],
  section[data-testid="stSidebar"] > div,
  section[data-testid="stSidebar"] > div > div {
    background: var(--cream) !important;
    border-right: none !important;
    box-shadow: none !important;
  }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {
    font-family: var(--serif) !important;
    color: var(--green-900) !important;
  }
  section[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.85rem !important;
    color: var(--text-muted) !important;
    line-height: 1.55 !important;
  }

  /* ── Metric cards (sidebar) ─────────────────────────── */
  .sidebar-metric {
    flex: 1;
  }
  .sidebar-metric-label {
    font-family: var(--sans);
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 0.15rem;
  }
  .sidebar-metric-value {
    font-family: var(--serif);
    font-size: 1.45rem;
    font-weight: 600;
    color: var(--green-900);
    line-height: 1.2;
  }
  .sidebar-metric-row {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 0.75rem;
  }

  /* ── Divider ────────────────────────────────────────── */
  hr {
    border: none !important;
    border-top: 1px solid var(--cream-dark) !important;
    margin: 1rem 0 !important;
  }

  /* ── Spinner / status ───────────────────────────────── */
  .stSpinner > div {
    border-top-color: var(--green-600) !important;
  }

  /* ── Buttons ────────────────────────────────────────── */
  .stButton > button {
    font-family: var(--sans) !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.03em;
    background: var(--green-900) !important;
    color: var(--cream) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.25rem !important;
    transition: background var(--transition), box-shadow var(--transition) !important;
  }
  .stButton > button:hover {
    background: var(--green-700) !important;
    box-shadow: var(--shadow-md) !important;
  }

  /* ── Scrollbar ──────────────────────────────────────── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb {
    background: var(--cream-dark);
    border-radius: 3px;
  }
  ::-webkit-scrollbar-thumb:hover { background: var(--warm-gray); }

  /* ── Animations ─────────────────────────────────────── */
  @keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  [data-testid="stChatMessage"] {
    animation: fadeSlideUp 0.35s ease-out !important;
  }

  /* ── Welcome card (custom component) ────────────────── */
  .welcome-card {
    background: white;
    border: 1px solid var(--cream-dark);
    border-radius: 14px;
    padding: 2.4rem 2rem 2rem;
    max-width: 620px;
    margin: 2.5rem auto 1rem;
    box-shadow: var(--shadow-md);
    text-align: center;
    animation: fadeSlideUp 0.5s ease-out;
  }
  .welcome-card .icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 48px; height: 48px;
    background: var(--green-50);
    border: 1px solid var(--green-100);
    border-radius: 12px;
    margin-bottom: 1.1rem;
    font-size: 1.3rem;
  }
  .welcome-card h2 {
    font-family: 'Fraunces', Georgia, serif !important;
    font-size: 1.55rem !important;
    color: var(--green-900) !important;
    margin: 0 0 0.45rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
  }
  .welcome-card .subtitle {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.88rem;
    color: var(--text-muted);
    margin-bottom: 1.6rem;
    line-height: 1.55;
  }

  .prompt-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 0.5rem;
  }
  .prompt-chip {
    background: var(--cream);
    border: 1px solid var(--cream-dark);
    border-radius: 10px;
    padding: 0.7rem 0.85rem;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.8rem;
    color: var(--text);
    text-align: left;
    line-height: 1.45;
    cursor: default;
    transition: border-color 0.2s, background 0.2s;
  }
  .prompt-chip:hover {
    border-color: var(--green-500);
    background: var(--green-50);
  }
  .prompt-chip .chip-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-light);
    margin-bottom: 0.3rem;
  }

  /* ── Prediction result card ─────────────────────────── */
  .result-card {
    background: linear-gradient(135deg, var(--green-50) 0%, white 100%);
    border: 1px solid var(--green-100);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin: 0.6rem 0;
  }
  .result-card .result-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 0.25rem;
  }
  .result-card .result-value {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--green-900);
  }
  .result-card .result-value.high-risk {
    color: var(--red-soft);
  }
  .result-card .result-value.medium-risk {
    color: var(--amber-soft);
  }

  /* ── Footer ─────────────────────────────────────────── */
  .app-footer {
    text-align: center;
    padding: 1.5rem 0 1rem;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.72rem;
    color: var(--text-light);
    letter-spacing: 0.02em;
  }
  .app-footer a {
    color: var(--green-600);
    text-decoration: none;
    font-weight: 500;
  }
  .app-footer a:hover { text-decoration: underline; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model Details")
    st.markdown("---")

    st.markdown(
        """
        <div class="sidebar-metric-row">
          <div class="sidebar-metric">
            <div class="sidebar-metric-label">AUC-ROC</div>
            <div class="sidebar-metric-value">0.678</div>
          </div>
          <div class="sidebar-metric">
            <div class="sidebar-metric-label">F1 Score</div>
            <div class="sidebar-metric-value">0.402</div>
          </div>
        </div>
        <div class="sidebar-metric-row">
          <div class="sidebar-metric">
            <div class="sidebar-metric-label">Precision</div>
            <div class="sidebar-metric-value">0.292</div>
          </div>
          <div class="sidebar-metric">
            <div class="sidebar-metric-label">Recall</div>
            <div class="sidebar-metric-value">0.645</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        "**Model** — XGBoost classifier trained on ~1.35M "
        "Lending Club loans, using application-time features only."
    )
    st.markdown(
        "**Threshold** — Tuned to 0.199 to balance recall "
        "for default detection against precision."
    )
    st.markdown(
        "**AUC ceiling** — 0.68 is the demonstrated ceiling for "
        "this feature set (Sanz-Guerrero & Arroyo, 2024)."
    )

    st.markdown("---")

    st.markdown("##### Required Features")
    st.markdown(
        "FICO score · Loan amount · Revenue"
    )
    st.markdown("##### Recommended Features")
    st.markdown(
        "Indebtedness (DTI) · Employment length · Loan purpose · Home ownership"
    )
    st.markdown("##### Optional")
    st.markdown("State · Prior borrowing experience")

    st.markdown("---")

    if st.button("✦  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        '<div class="app-footer">'
        'Built with Streamlit + XGBoost + GPT-5 mini<br>'
        '<a href="https://github.com/">View on GitHub</a>'
        '</div>',
        unsafe_allow_html=True,
    )


# ─── Session state ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ─── Welcome card (shown when conversation is empty) ────────────────
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-card">
          <div class="icon">◆</div>
          <h2>Credit Risk Advisor</h2>
          <p class="subtitle">
            Describe a loan applicant in plain English and the model will
            estimate their default probability. Powered by an XGBoost
            classifier trained on 1.35 million Lending Club loans.
          </p>
          <div class="prompt-grid">
            <div class="prompt-chip">
              <div class="chip-label">Try asking</div>
              FICO 710, requesting $15,000 for debt consolidation, earns $62K/year
            </div>
            <div class="prompt-chip">
              <div class="chip-label">Try asking</div>
              What's the risk for a 3-year loan of $8,000 with a 680 FICO?
            </div>
            <div class="prompt-chip">
              <div class="chip-label">Try asking</div>
              Low FICO around 630, high DTI of 28, $20K loan for a small business
            </div>
            <div class="prompt-chip">
              <div class="chip-label">Try asking</div>
              What features does the model need to make a prediction?
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_content(content, role):
    text = content.replace("$", r"\$") if role == "user" else content
    st.markdown(text, unsafe_allow_html=True)


# ─── Render conversation history ────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        display_content(msg["content"], msg["role"])


# ─── Chat input handling ────────────────────────────────────────────
if prompt := st.chat_input("Describe a loan applicant…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        display_content(prompt, "user")

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing…"):
            try:
                response = get_response(
                    prompt,
                    history=st.session_state.messages[:-1],
                )
            except EnvironmentError as e:
                response = f"⚠️ **Configuration error:** {e}"
            except Exception as e:
                response = f"⚠️ **Something went wrong:** {e}"

        st.markdown(response, unsafe_allow_html=True)
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )