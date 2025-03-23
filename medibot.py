import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
HF_TOKEN = os.environ.get("HF_TOKEN")

INSTRUCTION_TEXT = """
### 🚨 Important Instructions 🚨
📌 **MediBot is designed for medical assistance only**  
🔹 It provides **general medical advice, mental health support, and medicine-related guidance**.  
🔹 It **does not replace professional medical diagnosis**.  
🔹 For **emergency cases, consult a healthcare provider immediately**.  

🧠 **Mental Health Guidelines**:  
✔️ Stress, anxiety, and depression-related queries are supported.  
✔️ Tips on mindfulness, therapy, and self-care are available.  
❌ No detailed psychological analysis or crisis intervention.  

💊 **Medicine Information**:  
✔️ General usage, side effects, and interactions.  
❌ No prescription recommendations. Always consult a doctor.  
"""


CUSTOM_PROMPT_TEMPLATE = """
Use the provided context to answer the user's question accurately.
If unsure, say 'I don't know.' Do not generate false information.
Limit responses to the given medical context.

Context: {context}
Question: {question}

Answer concisely:
"""

# Cache vector store
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Load LLM model
def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

# Set prompt template
def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# Apply theme
def apply_theme(theme):
    themes = {
        "Dark": """
            <style>
            body { background-color: #0E1117; color: white; }
            .stButton>button { background-color: #1f77b4; color: white; }
            </style>
        """,
        "Light": """
            <style>
            body { background-color: white; color: black; }
            .stButton>button { background-color: #007bff; color: white; }
            </style>
        """,
    }
    st.markdown(themes.get(theme, ""), unsafe_allow_html=True)

def main():
    # Page Configuration
    st.set_page_config(page_title="MediBot - AI Medical Assistant", page_icon="🩺", layout="wide")
    
    # Sidebar settings
    with st.sidebar:
        st.image("https://via.placeholder.com/150", width=150, caption="MediBot AI")
        st.title("⚙️ Settings")
        num_results = st.slider("Number of documents to retrieve", 1, 10, 7)
        theme = st.selectbox("Choose Theme", ["Light", "Dark"])
        clear_chat = st.button("🗑️ Clear Chat History")
    
    apply_theme(theme)
    
    # Header
    st.markdown("""
        <style>
        .chat-title {text-align: center; font-size: 36px; font-weight: bold;}
        .chat-subtitle {text-align: center; font-size: 18px; color: gray;}
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='chat-title'>🩺 MediBot - AI Medical Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p class='chat-subtitle'>Ask about medical topics, and I'll assist you!</p>", unsafe_allow_html=True)
    
    # Chat History Management
    if clear_chat:
        st.session_state.messages = []
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        role_icon = "🧑" if message['role'] == 'user' else "🤖"
        st.chat_message(message['role']).markdown(f"{role_icon} {message['content']}")
    
    # User Input
    prompt = st.chat_input("Type your question here...")
    
    if prompt:
        st.chat_message('user').markdown(f"🧑 {prompt}")
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        try:
            with st.spinner("🤖 Thinking..."):
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("❌ Failed to load the vector store")
                    return
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': num_results}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt()}
                )
                
                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                
            st.chat_message('assistant').markdown(f"🤖 {result}")
            st.session_state.messages.append({'role': 'assistant', 'content': result})
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
