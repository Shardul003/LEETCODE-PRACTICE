# app.py - LangGraph Agent-Based RAG with Logging
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
from typing import TypedDict, Annotated, Sequence, Literal
from datetime import datetime
import operator
import streamlit as st
from dotenv import load_dotenv

# LangChain / models
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_core.tools import tool

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# PDF extraction
from pypdf import PdfReader

# -------------------------------
# LOGGING SETUP
# -------------------------------
def setup_logging():
    """Setup comprehensive logging system"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )
    
    loggers = {
        'main': logging.getLogger('main'),
        'agent': logging.getLogger('agent'),
        'tools': logging.getLogger('tools'),
        'vectorstore': logging.getLogger('vectorstore'),
        'chat': logging.getLogger('chat'),
        'performance': logging.getLogger('performance')
    }
    
    json_log_path = log_dir / f"chat_logs_{datetime.now().strftime('%Y%m%d')}.jsonl"
    
    return loggers, json_log_path

LOGGERS, JSON_LOG_PATH = setup_logging()

def log_chat_interaction(question: str, answer: str, tools_used: list, duration: float):
    """Log chat interaction in JSON format"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "tools_used": tools_used,
        "duration_seconds": duration,
        "top_k": st.session_state.get("top_k", 3)
    }
    
    with open(JSON_LOG_PATH, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    LOGGERS['chat'].info(f"Chat interaction logged - Duration: {duration:.2f}s, Tools: {tools_used}")

# -------------------------------
# ENV / CONFIG
# -------------------------------
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

DEFAULT_INDEX_DIR = str(Path.cwd() / "my_vectorstore123")

st.set_page_config(page_title="Budget Agent (LangGraph)", page_icon="🤖")

LOGGERS['main'].info("Agent application started")

# -------------------------------
# STATE DEFINITION
# -------------------------------
class AgentState(TypedDict):
    """State for the agent graph"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    question: str
    context: str
    answer: str
    tools_used: list
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
# DOCUMENT PROCESSING HELPERS
# -------------------------------
def extract_text_from_pdf_file(uploaded_file) -> list[Document]:
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
    LOGGERS['vectorstore'].info(f"Chunking {len(documents)} documents")
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
# AGENT TOOLS
# -------------------------------
@tool
def search_budget_documents(query: str) -> str:
    """Search the budget documents for relevant information. Use this tool to find budget-related data, 
    allocations, expenditures, or any information from the loaded documents."""
    LOGGERS['tools'].info(f"Tool called: search_budget_documents with query: {query[:100]}")
    
    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is None:
        LOGGERS['tools'].warning("No vectorstore available")
        return "Error: No documents have been loaded. Please upload and index documents first."
    
    k = st.session_state.get("top_k", 3)
    
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(query)
        
        LOGGERS['tools'].info(f"Retrieved {len(docs)} documents")
        
        if not docs:
            return "No relevant documents found for this query."
        
        # Format results
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', 'N/A')
            content = doc.page_content[:500]  # Limit content length
            results.append(f"[Document {i}] Source: {source}, Page: {page}\n{content}...")
        
        return "\n\n".join(results)
    
    except Exception as e:
        LOGGERS['tools'].error(f"Error in search_budget_documents: {e}")
        return f"Error searching documents: {str(e)}"

@tool
def list_loaded_sources() -> str:
    """List all documents that have been loaded into the knowledge base."""
    LOGGERS['tools'].info("Tool called: list_loaded_sources")
    
    sources = st.session_state.get("loaded_sources", set())
    if not sources:
        return "No documents have been loaded yet."
    
    return "Loaded documents:\n" + "\n".join(f"- {s}" for s in sorted(sources))

@tool
def calculate_budget_total(category: str) -> str:
    """Calculate total budget amounts. This is a placeholder - extend with actual calculation logic."""
    LOGGERS['tools'].info(f"Tool called: calculate_budget_total for category: {category}")
    
    return f"Budget calculation tool called for category: {category}. Note: This is a placeholder. Implement actual calculation logic based on your needs."

# -------------------------------
# AGENT NODES
# -------------------------------
def agent_node(state: AgentState) -> AgentState:
    """The main agent that decides what to do next"""
    LOGGERS['agent'].info("Agent node processing")
    
    messages = state.get("messages", [])
    
    # Get LLM with tools
    llm = get_llm()
    tools = [search_budget_documents, list_loaded_sources, calculate_budget_total]
    llm_with_tools = llm.bind_tools(tools)
    
    # Create system message
    system_msg = SystemMessage(content="""You are a helpful budget assistant agent. You have access to tools to search budget documents and provide accurate information.

Use the search_budget_documents tool to find relevant information from the loaded documents.
Use the list_loaded_sources tool to see what documents are available.
Use the calculate_budget_total tool for budget calculations (placeholder).

Always cite your sources and be precise. If you don't know something, say so.""")
    
    # Invoke the agent
    messages_with_system = [system_msg] + list(messages)
    response = llm_with_tools.invoke(messages_with_system)
    
    LOGGERS['agent'].info(f"Agent response type: {type(response)}")
    
    return {"messages": [response]}

def tool_node_func(state: AgentState) -> AgentState:
    """Execute tools requested by the agent"""
    LOGGERS['agent'].info("Tool node processing")
    
    messages = state.get("messages", [])
    last_message = messages[-1]
    
    tool_calls = getattr(last_message, "tool_calls", [])
    LOGGERS['tools'].info(f"Executing {len(tool_calls)} tool calls")
    
    tools_used = []
    tool_outputs = []
    
    tools_map = {
        "search_budget_documents": search_budget_documents,
        "list_loaded_sources": list_loaded_sources,
        "calculate_budget_total": calculate_budget_total
    }
    
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tools_used.append(tool_name)
        
        LOGGERS['tools'].info(f"Executing tool: {tool_name}")
        
        if tool_name in tools_map:
            result = tools_map[tool_name].invoke(tool_args)
            tool_outputs.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                )
            )
    
    return {"messages": tool_outputs, "tools_used": tools_used}

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Decide whether to continue with tools or end"""
    messages = state.get("messages", [])
    last_message = messages[-1]
    
    # If there are tool calls, continue to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        LOGGERS['agent'].info("Continuing to tools")
        return "tools"
    
    # Otherwise, end
    LOGGERS['agent'].info("Ending agent loop")
    return "end"

def final_response_node(state: AgentState) -> AgentState:
    """Generate final response to user"""
    LOGGERS['agent'].info("Generating final response")
    
    messages = state.get("messages", [])
    
    # The last message should be the agent's response
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage):
            answer = last_msg.content
        else:
            answer = "I apologize, but I couldn't generate a proper response."
    else:
        answer = "No response generated."
    
    # Log interaction
    start_time = state.get("start_time", datetime.now().timestamp())
    duration = datetime.now().timestamp() - start_time
    tools_used = state.get("tools_used", [])
    question = state.get("question", "")
    
    log_chat_interaction(question, answer, tools_used, duration)
    
    return {"answer": answer}

# -------------------------------
# BUILD LANGGRAPH AGENT
# -------------------------------
@st.cache_resource(show_spinner=False)
def build_agent_graph():
    """Build the LangGraph agent workflow"""
    LOGGERS['main'].info("Building LangGraph agent workflow")
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node_func)
    workflow.add_node("final_response", final_response_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": "final_response"
        }
    )
    
    # After tools, go back to agent
    workflow.add_edge("tools", "agent")
    
    # Final response leads to END
    workflow.add_edge("final_response", END)
    
    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    LOGGERS['main'].info("LangGraph agent workflow built successfully")
    return app

# -------------------------------
# SESSION STATE
# -------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm your budget assistant agent. Upload PDFs and I'll help you analyze them using my tools."}
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
    st.session_state["thread_id"] = "budget_agent_1"
if "interaction_count" not in st.session_state:
    st.session_state["interaction_count"] = 0

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("🤖 Agent Configuration")
    
    # Agent status
    with st.expander("📊 Agent Status", expanded=False):
        st.success("✅ Agent Active")
        st.caption("**Available Tools:**")
        st.caption("🔍 search_budget_documents")
        st.caption("📄 list_loaded_sources")
        st.caption("🧮 calculate_budget_total")
        st.divider()
        st.caption(f"Logs: `logs/app_{datetime.now().strftime('%Y%m%d')}.log`")
        st.caption(f"Interactions: {st.session_state['interaction_count']}")
    
    st.divider()
    st.subheader("📚 Knowledge Base")
    
    index_dir = st.text_input("Index directory", value=st.session_state["last_index_dir"])
    st.session_state["last_index_dir"] = index_dir

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
    persist = st.checkbox("Persist index to disk", value=True)

    colA, colB, colC = st.columns(3)
    with colA:
        build_btn = st.button("Build/Update", use_container_width=True)
    with colB:
        load_btn = st.button("Load Index", use_container_width=True)
    with colC:
        clear_btn = st.button("Clear", use_container_width=True)

    st.divider()
    k = st.slider("Top-K documents", 1, 10, 3)
    st.session_state["top_k"] = k

# -------------------------------
# INDEX MANAGEMENT
# -------------------------------
embeddings = get_embeddings()

if build_btn:
    if not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        with st.spinner("Building index..."):
            all_docs = []
            for f in uploaded_files:
                if f.name.lower().endswith(".pdf"):
                    all_docs.extend(extract_text_from_pdf_file(f))
                else:
                    all_docs.extend(extract_text_from_txt_file(f))

            if not all_docs:
                st.error("No text extracted.")
            else:
                chunks = chunk_documents(all_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                if build_mode == "Replace index" or st.session_state["vectorstore"] is None:
                    vs = build_faiss_from_docs(chunks, embeddings)
                else:
                    vs = st.session_state["vectorstore"]
                    vs.add_documents(chunks)

                st.session_state["vectorstore"] = vs
                for d in all_docs:
                    st.session_state["loaded_sources"].add(d.metadata.get("source", "unknown"))

                if persist:
                    save_index(st.session_state["vectorstore"], index_dir)
                    st.success(f"✅ Index saved to: {index_dir}")
                else:
                    st.info("Index updated in memory.")

if load_btn:
    try:
        with st.spinner("Loading index..."):
            st.session_state["vectorstore"] = load_index(index_dir, embeddings)
            st.success(f"✅ Loaded from: {index_dir}")
    except Exception as e:
        st.error(f"❌ Failed: {e}")

if clear_btn:
    st.session_state["vectorstore"] = None
    st.session_state["loaded_sources"] = set()
    st.success("🧹 Cleared.")
    st.rerun()

# -------------------------------
# MAIN APP
# -------------------------------
st.title("🤖 Budget Assistant Agent")
st.caption("Powered by LangGraph • Agentic RAG with Tool Use")

# Show loaded sources
if st.session_state["loaded_sources"]:
    st.caption("📄 " + " | ".join(sorted(st.session_state["loaded_sources"])))
else:
    st.caption("No documents loaded.")

# Display chat
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
if user_input := st.chat_input("Ask me anything about the budget..."):
    LOGGERS['chat'].info(f"User question: {user_input}")
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent thinking..."):
            # Build agent graph
            agent_graph = build_agent_graph()
            
            # Convert messages to LangChain format
            lc_messages = []
            for msg in st.session_state["messages"]:
                if msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    lc_messages.append(AIMessage(content=msg["content"]))
            
            # Prepare state
            initial_state = {
                "messages": lc_messages,
                "question": user_input,
                "context": "",
                "answer": "",
                "tools_used": [],
                "start_time": datetime.now().timestamp()
            }
            
            config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
            
            try:
                # Run agent
                result = agent_graph.invoke(initial_state, config)
                answer = result.get("answer", "I couldn't generate a response.")
                
                st.session_state["interaction_count"] += 1
                LOGGERS['chat'].info(f"Agent completed (interaction #{st.session_state['interaction_count']})")
                
                # Show tools used
                tools_used = result.get("tools_used", [])
                if tools_used:
                    st.caption(f"🔧 Tools used: {', '.join(tools_used)}")
                
            except Exception as e:
                answer = f"Error: {e}"
                LOGGERS['chat'].error(f"Agent error: {e}", exc_info=True)
            
            st.session_state["messages"].append({"role": "assistant", "content": answer})
            st.write(answer)

# Utility buttons
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("♻️ Reset"):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Chat reset!"}
        ]
        st.session_state["thread_id"] = f"agent_{datetime.now().timestamp()}"
        st.rerun()
with c2:
    if st.button("🧹 Clear Index"):
        st.session_state["vectorstore"] = None
        st.session_state["loaded_sources"] = set()
        st.rerun()
with c3:
    if st.button("📊 Logs"):
        st.info("Check `logs/` folder")
with c4:
    if st.button("📈 Stats"):
        st.metric("Interactions", st.session_state["interaction_count"])