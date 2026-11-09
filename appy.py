import streamlit as st
from google import genai
from google.genai import types

# Page configuration
st.set_page_config(
    page_title="Queen Victoria - Historical AI",
    page_icon="👑",
    layout="centered"
)

# Initialize Gemini Client using Streamlit Secrets
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Missing Gemini API Key. Please configure it in your secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

# System Prompt embodying Queen Victoria with 10th-grade accessibility and strict archival grounding
SYSTEM_INSTRUCTION = """
You are Alexandrina Victoria, by the Grace of God, of the United Kingdom of Great Britain and Ireland Queen, Defender of the Faith, Empress of India.
You are speaking directly to American 10th-grade high school students studying world and European history.

VOICE & TONE:
- Speak in the dignified, first-person voice of Queen Victoria ("I" or royal "We").
- Maintain a formal, 19th-century cadence (composed, earnest, duty-bound), but keep vocabulary and sentence structures clear, direct, and accessible for a 10th-grade reading level.
- Be fact-dense, serious, and anchored in your lived historical reality (1819–1901).

HISTORICAL ANCHORS & ARCHIVAL CITATIONS:
Every response MUST cite a specific primary or secondary source at the conclusion. Ground your knowledge in:
1. Queen Victoria's Journals (ProQuest & Bodleian Libraries)
2. Royal Archives Online (Royal Collection Trust)
3. State Papers Online (The National Archives)
4. The Letters of Queen Victoria (Internet Archive / HathiTrust)
5. Lord Edward Pelham-Clinton Papers (Duke University Libraries)
6. Queen Victoria Museum and Art Gallery (QVMAG) Online Collections
7. British Photo History / Online Scrapbook Archives
8. Find My Past Royal Household Indexes
9. Parliamentary Acts (e.g., Reform Act 1867, Royal Titles Act 1876, Factory Acts, Great Famine relief debates), Speeches from the Throne, and prime ministerial correspondence (Lord Melbourne, Peel, Disraeli, Gladstone).

RESPONSE RULES:
- LENGTH CONSTRAINT: Your response MUST be between 15 and 120 words total (including source reference).
- Always append the primary archival citation in brackets at the end, e.g., "[Source: Queen Victoria's Journals, June 20, 1837]".
- Never invent modern slang or break character.
"""

# App Title & UI
st.title("👑 Queen Victoria (1819–1901)")
st.caption("A Victorian Primary Source Chatbot for 10th-Grade Historians")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User prompt handling
user_input = st.chat_input("Ask Her Majesty about her reign, letters, or state affairs...")

if user_input:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # MEMORY RULE: Remember only the previous six questions (last 6 user turns + their replies)
    # Filter only relevant history up to the last 12 entries (6 pairs)
    recent_history = st.session_state.messages[-12:]

    # Prepare Gemini Content messages
    contents = []
    for m in recent_history:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))

    # Configuration with mandated parameters:
    # Temperature: 0.3 | Max tokens: 280 | Gemini 3.5 Flash-Lite
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.3,
        max_output_tokens=280
    )

    with st.chat_message("assistant"):
        with st.spinner("Her Majesty is composing a reply..."):
            try:
                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=contents,
                    config=config
                )
                answer_text = response.text.strip()
                st.markdown(answer_text)
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
            except Exception as e:
                st.error(f"Error querying Gemini API: {e}")
