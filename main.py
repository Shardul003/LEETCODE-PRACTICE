# app.py - LangGraph Version with Logging
# 
# Installation requirements (requirements.txt):
# langchain==0.3.13
# langchain-community==0.3.13
# langchain-core==0.3.28
# langchain-groq==0.2.1
# langchain-huggingface==0.1.2
# langchain-text-splitters==0.3.4
# langgraph==0.2.58
# langgraph-checkpoint==2.0.9
# streamlit==1.41.1
# python-dotenv==1.0.1
# pypdf==5.1.0
# faiss-cpu==1.9.0.post1
# sentence-transformers==3.3.1
# huggingface-hub==0.27.0
# groq==0.13.0

import os
import logging
import json
from io import BytesIO
from pathlib import Path
from typing import TypedDict, Annotated, Sequence
from datetime import datetime
import operator
import streamlit as st
from dotenv import load_dotenv

# LangChain / models
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# PDF extraction
from pypdf import PdfReader

# -------------------------------
# LOGGING SETUP
# -------------------------------
def setup_logging():
    """Setup comprehensive logging system"""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create specialized loggers
    loggers = {
        'main': logging.getLogger('main'),
        'rag': logging.getLogger('rag'),
        'vectorstore': logging.getLogger('vectorstore'),
        'chat': logging.getLogger('chat'),
        'performance': logging.getLogger('performance')
    }
    
    # Create JSON log file for structured data
    json_log_path = log_dir / f"chat_logs_{datetime.now().strftime('%Y%m%d')}.jsonl"
    
    return loggers, json_log_path

# Initialize logging
LOGGERS, JSON_LOG_PATH = setup_logging()

def log_chat_interaction(question: str, answer: str, context: str, sources: list, duration: float):
    """Log chat interaction in JSON format"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "context_length": len(context),
        "sources": sources,
        "duration_seconds": duration,
        "top_k": st.session_state.get("top_k", 3)
    }
    
    with open(JSON_LOG_PATH, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    LOGGERS['chat'].info(f"Chat interaction logged - Duration: {duration:.2f}s")

# -------------------------------
# ENV / CONFIG
# -------------------------------
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

DEFAULT_INDEX_DIR = str(Path.cwd() / "my_vectorstore123")

st.set_page_config(page_title="Budget Chatbot (LangGraph)", page_icon="💬")

LOGGERS['main'].info("Application started")

# -------------------------------
# STATE DEFINITION
# -------------------------------
class GraphState(TypedDict):
    """State for the RAG graph"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    question: str
    context: str
    answer: str
    retrieved_docs: list
    start_time: float

# -------------------------------
# CACHED RESOURCES
# -------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    LOGGERS['main'].info("Loading embeddings model: all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    LOGGERS['main'].info("Embeddings model loaded successfully")
    return embeddings

@st.cache_resource(show_spinner=False)
def get_llm():
    LOGGERS['main'].info("Initializing Groq LLM: llama-3.1-8b-instant")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    LOGGERS['main'].info("LLM initialized successfully")
    return llm

# -------------------------------
# HELPERS
# -------------------------------
def extract_text_from_pdf_file(uploaded_file) -> list[Document]:
    """Extract text from a Streamlit UploadedFile using pypdf"""
    LOGGERS['vectorstore'].info(f"Extracting text from PDF: {uploaded_file.name}")
    name = uploaded_file.name
    file_bytes = uploaded_file.read()
    reader = PdfReader(BytesIO(file_bytes))
    docs = []
    
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        if txt.strip():
            docs.append(
                Document(
                    page_content=txt,
                    metadata={"source": name, "page": i + 1}
                )
            )
    
    LOGGERS['vectorstore'].info(f"Extracted {len(docs)} pages from {name}")
    return docs

def extract_text_from_txt_file(uploaded_file) -> list[Document]:
    LOGGERS['vectorstore'].info(f"Extracting text from TXT: {uploaded_file.name}")
    content = uploaded_file.read().decode("utf-8", errors="ignore")
    doc = Document(page_content=content, metadata={"source": uploaded_file.name})
    LOGGERS['vectorstore'].info(f"Extracted {len(content)} characters from {uploaded_file.name}")
    return [doc]

def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    LOGGERS['vectorstore'].info(f"Chunking {len(documents)} documents (size={chunk_size}, overlap={chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    LOGGERS['vectorstore'].info(f"Created {len(chunks)} chunks")
    return chunks

def build_faiss_from_docs(docs, embeddings):
    LOGGERS['vectorstore'].info(f"Building FAISS index from {len(docs)} documents")
    start = datetime.now()
    vectorstore = FAISS.from_documents(docs, embeddings)
    duration = (datetime.now() - start).total_seconds()
    LOGGERS['performance'].info(f"FAISS index built in {duration:.2f} seconds")
    return vectorstore

def save_index(vectorstore, index_dir):
    LOGGERS['vectorstore'].info(f"Saving index to {index_dir}")
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(index_dir)
    LOGGERS['vectorstore'].info("Index saved successfully")

def load_index(index_dir, embeddings):
    LOGGERS['vectorstore'].info(f"Loading index from {index_dir}")
    vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    LOGGERS['vectorstore'].info("Index loaded successfully")
    return vectorstore

# -------------------------------
# LANGGRAPH NODES
# -------------------------------
def retrieve_documents(state: GraphState) -> GraphState:
    """Retrieve relevant documents from vectorstore"""
    start_time = datetime.now()
    question = state["question"]
    vectorstore = st.session_state.get("vectorstore")
    k = st.session_state.get("top_k", 3)
    
    LOGGERS['rag'].info(f"Retrieving documents for question: {question[:100]}...")
    
    if vectorstore is None:
        LOGGERS['rag'].warning("No vectorstore loaded")
        return {"context": "No vectorstore loaded.", "retrieved_docs": [], "start_time": start_time.timestamp()}
    
    # Retrieve documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(question)
    
    LOGGERS['rag'].info(f"Retrieved {len(docs)} documents (k={k})")
    
    # Log retrieved sources
    sources = [doc.metadata.get('source', 'unknown') for doc in docs]
    LOGGERS['rag'].debug(f"Sources: {sources}")
    
    # Format context
    context = "\n\n".join([f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}" 
                           for doc in docs])
    
    retrieval_time = (datetime.now() - start_time).total_seconds()
    LOGGERS['performance'].info(f"Document retrieval took {retrieval_time:.2f} seconds")
    
    return {
        "context": context,
        "retrieved_docs": [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in docs],
        "start_time": start_time.timestamp()
    }

def generate_answer(state: GraphState) -> GraphState:
    """Generate answer using LLM based on context"""
    start_time = datetime.now()
    question = state["question"]
    context = state["context"]
    messages = state.get("messages", [])
    
    LOGGERS['rag'].info(f"Generating answer for question: {question[:100]}...")
    
    llm = get_llm()
    
    # Create prompt with chat history
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful budget assistant. Use ONLY the provided context to answer questions.
If the answer is not in the context, say "I don't know based on the provided documents."
Be concise and accurate. Cite sources when possible."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", """Context:
{context}

Question: {question}

Answer:""")
    ])
    
    # Get chat history (excluding current question)
    chat_history = messages[:-1] if len(messages) > 1 else []
    
    LOGGERS['rag'].debug(f"Chat history length: {len(chat_history)}")
    
    # Generate answer
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history
    })
    
    answer = response.content
    generation_time = (datetime.now() - start_time).total_seconds()
    
    LOGGERS['rag'].info(f"Answer generated (length: {len(answer)} chars)")
    LOGGERS['performance'].info(f"Answer generation took {generation_time:.2f} seconds")
    
    # Log complete interaction
    total_time = datetime.now().timestamp() - state.get("start_time", datetime.now().timestamp())
    sources = [doc['source'] for doc in state.get('retrieved_docs', [])]
    log_chat_interaction(question, answer, context, sources, total_time)
    
    return {"answer": answer}

# -------------------------------
# BUILD LANGGRAPH
# -------------------------------
@st.cache_resource(show_spinner=False)
def build_graph():
    """Build the LangGraph workflow"""
    LOGGERS['main'].info("Building LangGraph workflow")
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    # Add edges
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    LOGGERS['main'].info("LangGraph workflow built successfully")
    return app

# -------------------------------
# SESSION STATE
# -------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! Upload PDFs and ask me anything about the budget documents."}
    ]
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "last_index_dir" not in st.session_state:
    st.session_state["last_index_dir"] = DEFAULT_INDEX_DIR
if "loaded_sources" not in st.session_state:
    st.session_state["loaded_sources"] = set()
if "top_k" not in st.session_state:
    st.session_state["top_k"] = 3
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = "budget_chat_1"
if "interaction_count" not in st.session_state:
    st.session_state["interaction_count"] = 0

# -------------------------------
# SIDEBAR (Index Management)
# -------------------------------
with st.sidebar:
    st.header("📚 Knowledge Base")
    
    # Logging status indicator
    with st.expander("📊 Logging Status", expanded=False):
        st.success("✅ Logging Active")
        st.caption(f"Logs: `logs/app_{datetime.now().strftime('%Y%m%d')}.log`")
        st.caption(f"Chat logs: `logs/chat_logs_{datetime.now().strftime('%Y%m%d')}.jsonl`")
        st.caption(f"Total interactions: {st.session_state['interaction_count']}")
    
    st.divider()
    
    index_dir = st.text_input("Index directory", value=st.session_state["last_index_dir"])
    st.session_state["last_index_dir"] = index_dir

    # Upload controls
    uploaded_files = st.file_uploader(
        "Upload PDF/TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    build_mode = st.radio(
        "Build mode",
        options=["Replace index", "Append to current"],
        help="Replace will discard the current vectorstore; Append adds new chunks."
    )
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=3000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=100, step=10)
    persist = st.checkbox("Persist index to disk after build", value=True)

    colA, colB, colC = st.columns(3)
    with colA:
        build_btn = st.button("Build/Update", use_container_width=True)
    with colB:
        load_btn = st.button("Load Index", use_container_width=True)
    with colC:
        clear_btn = st.button("Clear Index", use_container_width=True)

    st.divider()
    k = st.slider("Top-K documents", 1, 10, 3)
    st.session_state["top_k"] = k
    st.caption("Tip: Larger K can improve recall but may slow responses.")

# -------------------------------
# BUILD / LOAD / CLEAR INDEX ACTIONS
# -------------------------------
embeddings = get_embeddings()

if build_btn:
    if not uploaded_files:
        st.warning("Please upload at least one PDF/TXT file.")
        LOGGERS['main'].warning("Build attempted without files")
    else:
        LOGGERS['main'].info(f"Building index from {len(uploaded_files)} files")
        with st.spinner("Building index from uploaded files..."):
            all_docs = []
            for f in uploaded_files:
                if f.name.lower().endswith(".pdf"):
                    all_docs.extend(extract_text_from_pdf_file(f))
                else:
                    all_docs.extend(extract_text_from_txt_file(f))

            if not all_docs:
                st.error("No extractable text found in the uploaded files.")
                LOGGERS['vectorstore'].error("No text extracted from uploaded files")
            else:
                chunks = chunk_documents(all_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                if build_mode == "Replace index" or st.session_state["vectorstore"] is None:
                    vs = build_faiss_from_docs(chunks, embeddings)
                    LOGGERS['vectorstore'].info("Created new index")
                else:
                    vs = st.session_state["vectorstore"]
                    vs.add_documents(chunks)
                    LOGGERS['vectorstore'].info(f"Added {len(chunks)} chunks to existing index")

                st.session_state["vectorstore"] = vs
                for d in all_docs:
                    st.session_state["loaded_sources"].add(d.metadata.get("source", "unknown"))

                if persist:
                    save_index(st.session_state["vectorstore"], index_dir)
                    st.success(f"✅ Index saved to: {index_dir}")
                else:
                    st.info("Index updated in memory (not persisted).")

if load_btn:
    try:
        with st.spinner(f"Loading index from {index_dir}..."):
            st.session_state["vectorstore"] = load_index(index_dir, embeddings)
            st.success(f"✅ Loaded index from: {index_dir}")
    except Exception as e:
        st.error(f"❌ Failed to load index: {e}")
        LOGGERS['vectorstore'].error(f"Failed to load index: {e}")

if clear_btn:
    st.session_state["vectorstore"] = None
    st.session_state["loaded_sources"] = set()
    st.success("🧹 Cleared in-memory index.")
    LOGGERS['vectorstore'].info("Index cleared from memory")

# -------------------------------
# MAIN APP
# -------------------------------
st.title("💬 Budget Chatbot (LangGraph)")

# Show loaded sources
if st.session_state["loaded_sources"]:
    st.caption("📄 Loaded: " + " | ".join(sorted(st.session_state["loaded_sources"])))
elif st.session_state["vectorstore"] is not None:
    st.caption("Vectorstore loaded.")
else:
    st.caption("No index loaded yet.")

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Guardrail if no index
if st.session_state["vectorstore"] is None:
    st.info("⬆️ Upload and build an index (or load an existing one) to start chatting.")
    user_input = st.chat_input("Ask a question...")
    if user_input:
        st.chat_message("user").write(user_input)
        st.chat_message("assistant").write("Please build or load an index first.")
        LOGGERS['chat'].warning("Chat attempted without vectorstore")
    st.stop()

# -------------------------------
# CHAT WITH LANGGRAPH
# -------------------------------
graph_app = build_graph()

if user_input := st.chat_input("Ask a question..."):
    LOGGERS['chat'].info(f"User question: {user_input}")
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            # Convert messages to LangChain format
            lc_messages = []
            for msg in st.session_state["messages"]:
                if msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                else:
                    lc_messages.append(AIMessage(content=msg["content"]))
            
            # Prepare initial state
            initial_state = {
                "messages": lc_messages,
                "question": user_input,
                "context": "",
                "answer": "",
                "retrieved_docs": [],
                "start_time": datetime.now().timestamp()
            }
            
            # Run the graph
            config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
            
            try:
                result = graph_app.invoke(initial_state, config)
                answer = result.get("answer", "I don't know.")
                st.session_state["interaction_count"] += 1
                LOGGERS['chat'].info(f"Answer generated successfully (interaction #{st.session_state['interaction_count']})")
            except Exception as e:
                answer = f"Error: {e}"
                LOGGERS['chat'].error(f"Error generating answer: {e}", exc_info=True)
            
            st.session_state["messages"].append({"role": "assistant", "content": answer})
            st.write(answer)

# Utility buttons
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("♻️ Reset Chat"):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Chat reset. Ask me anything!"}
        ]
        st.session_state["thread_id"] = f"budget_chat_{datetime.now().timestamp()}"
        LOGGERS['chat'].info("Chat reset")
        st.rerun()
with c2:
    if st.button("🧹 Clear Index"):
        st.session_state["vectorstore"] = None
        st.session_state["loaded_sources"] = set()
        LOGGERS['vectorstore'].info("Index cleared")
        st.rerun()
with c3:
    if st.button("📊 View Logs"):
        st.info(f"Check logs folder: `logs/`")
with c4:
    if st.button("📈 Stats"):
        st.metric("Total Interactions", st.session_state["interaction_count"])