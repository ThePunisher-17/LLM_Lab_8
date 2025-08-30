import streamlit as st
import os
import time
import sys, asyncio
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Optional: Groq SDK for preflight auth check
try:
    import groq as groq_sdk
except Exception:
    groq_sdk = None

# -------------------- Streamlit / Page Config --------------------
st.set_page_config(page_title="LawGPT", page_icon="⚖️", layout="wide")

st.markdown("""
    <style>
    div.stButton > button:first-child { background-color: #ffd0d0; }
    div.stButton > button:active { background-color: #ff6262; }
    div[data-testid="stStatusWidget"] div button { display: none; }
    .reportview-container { margin-top: -2em; }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    button[title="View fullscreen"] { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# -------------------- Asyncio loop for Windows gRPC --------------------
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# -------------------- Load env / secrets --------------------
load_dotenv(override=True)
local_env = Path(__file__).with_name(".env")
if local_env.exists():
    load_dotenv(local_env, override=True)

# Read keys from environment/secrets only (no UI)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")

# -------------------- Title --------------------
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.title("Legal Information Chatbot")

# -------------------- Sidebar: ONLY Tools & Technologies --------------------
with st.sidebar:
    st.subheader("🛠️ Tools & Technologies")
    st.markdown(
        """
-- **Laws**: Criminal, Labour, Companies and Copyright laws 
- **LLM**: Llama-3 70B 
- **Framework**: Langchain
- **Embeddings**: Google Generative AI (`models/embedding-001`)
- **Vector Store**: FAISS 
- **Memory**: ConversationBufferWindowMemory `(k=2)`
- **Loaders**: PyPDFDirectoryLoader
- **Chunking**: RecursiveCharacterTextSplitter `(1000, overlap=200)`
- **Multilingual**: `langdetect` 
        """
    )

# -------------------- Hard checks for keys --------------------
errors = []
if not os.environ.get("GOOGLE_API_KEY"):
    errors.append("Missing GOOGLE_API_KEY in environment or Streamlit secrets")
if not groq_api_key:
    errors.append("Missing GROQ_API_KEY in environment or Streamlit secrets")
elif not groq_api_key.startswith("gsk_"):
    errors.append("GROQ_API_KEY format looks wrong (should start with 'gsk_').")

if errors:
    st.error(" / ".join(errors))
    st.stop()

# -------------------- Optional preflight against Groq --------------------
if groq_sdk is not None:
    try:
        groq_client = groq_sdk.Groq(api_key=groq_api_key)
        _ = groq_client.models.list()
    except Exception as e:
        if e.__class__.__name__ == "AuthenticationError" or "invalid_api_key" in str(e).lower():
            st.error("Groq rejected your API key (invalid/expired).")
            st.stop()
        st.error(f"Groq key check failed: {e}")
        st.stop()

# -------------------- Session state --------------------
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=2, memory_key="chat_history", return_messages=True
    )

# -------------------- Multilingual Support --------------------
try:
    from langdetect import detect
except Exception:
    detect = None

LANG_CHOICES = [
    "Auto-detect",
    "English",
    "हिन्दी (Hindi)",
    "தமிழ் (Tamil)",
    "తెలుగు (Telugu)",
    "ಕನ್ನಡ (Kannada)",
    "മലയാളം (Malayalam)",
    "मराठी (Marathi)",
    "বাংলা (Bengali)",
    "ગુજરાતી (Gujarati)",
    "ਪੰਜਾਬੀ (Punjabi)",
    "ଓଡ଼ିଆ (Odia)",
    "اردو (Urdu)",
]
LANG_NAME_CANON = {
    "Auto-detect": None,
    "English": "English",
    "हिन्दी (Hindi)": "Hindi",
    "தமிழ் (Tamil)": "Tamil",
    "తెలుగు (Telugu)": "Telugu",
    "ಕನ್ನಡ (Kannada)": "Kannada",
    "മലയാളം (Malayalam)": "Malayalam",
    "मराठी (Marathi)": "Marathi",
    "বাংলা (Bengali)": "Bengali",
    "ગુજરાતી (Gujarati)": "Gujarati",
    "ਪੰਜਾਬੀ (Punjabi)": "Punjabi",
    "ଓଡ଼ିଆ (Odia)": "Odia",
    "اردو (Urdu)": "Urdu",
}
CODE_TO_NAME = {
    "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
    "ml": "Malayalam", "mr": "Marathi", "bn": "Bengali", "gu": "Gujarati",
    "pa": "Punjabi", "or": "Odia", "ur": "Urdu"
}

with st.container():
    left, right = st.columns([3, 3])
    with left:
        lang_choice = st.selectbox(
            "Answer language",
            options=LANG_CHOICES,
            index=0,
            help="Choose a target language. Auto-detect uses your message language."
        )
    # (removed any UI text about keys being loaded)

def _augment_question_with_language(user_text: str, choice: str):
    """Returns (augmented_question, resolved_language_name)"""
    if choice != "Auto-detect":
        lang_name = LANG_NAME_CANON[choice]
        return f"Language: {lang_name}\n{user_text}", lang_name

    if detect is not None:
        try:
            code = detect(user_text)
            lang_name = CODE_TO_NAME.get(code, "English")
            return f"Language: {lang_name}\n{user_text}", lang_name
        except Exception:
            pass
    return user_text, "Auto"

# -------------------- Embeddings / FAISS --------------------
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.load_local("my_vector_store", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# -------------------- Prompt with language directive --------------------
prompt_template = """
<s>[INST]You are a legal chatbot. Provide accurate, concise answers grounded in the retrieved CONTEXT.
Do NOT fabricate. If information is not present in CONTEXT, say you don't have enough context and provide a brief general answer if appropriate.

**Language policy**
- If the QUESTION begins with a line like "Language: <Name>", respond in that language (e.g., Hindi, Tamil, Telugu, etc.).
- Otherwise, respond in the same language as the QUESTION.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}

ANSWER:
</s>[INST]
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])

# -------------------- LLM --------------------
try:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
except Exception as e:
    msg = str(e)
    if "invalid_api_key" in msg.lower() or "authenticationerror" in e.__class__.__name__.lower():
        st.error("GROQ_API_KEY is invalid.")
        st.stop()
    raise

# -------------------- Chain --------------------
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# -------------------- Chat History --------------------
for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

# -------------------- Input & Inference --------------------
input_prompt = st.chat_input("Type your question (supports Indian languages)…")

if input_prompt:
    augmented_question, resolved_lang = _augment_question_with_language(input_prompt, lang_choice)

    with st.chat_message("user"):
        lang_tag = f"  _(Language: {resolved_lang})_" if resolved_lang != "Auto" else ""
        st.write(input_prompt + lang_tag)

    st.session_state.messages.append({"role": "user", "content": input_prompt})

    with st.chat_message("assistant"):
        with st.status("Thinking 💡...", expanded=True):
            result = qa.invoke({"question": augmented_question})
            message_placeholder = st.empty()
            full_response = "\n\n\n"
            for ch in result["answer"]:
                full_response += ch
                time.sleep(0.02)
                message_placeholder.markdown(full_response + " ▌")

        st.button('Reset All Chat 🗑️', on_click=reset_conversation)

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
