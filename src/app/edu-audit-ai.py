"""
EDU-Audit Streamlit Chat Interface
"""

import streamlit as st
import requests
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[2] / '.env.dev'
load_dotenv(env_path)

BASE_URL = "http://localhost:8000"

# ── API 래핑 함수 ──
def upload_document(file_path: str):
    with open(file_path, "rb") as f:
        res = requests.post(f"{BASE_URL}/document/upload", files={"file": f})
    return res.json()

def list_documents(_=""):
    res = requests.get(f"{BASE_URL}/document/list")
    return res.json()

def analyze_full(doc_id: str):
    res = requests.post(f"{BASE_URL}/document/{doc_id}/analyze/full")
    return res.json()


# ── LangChain Agent ──
tools = [
    Tool(name="upload_document", func=upload_document,
         description="문서를 업로드합니다. 입력값은 파일 경로입니다."),
    Tool(name="list_documents", func=list_documents,
         description="업로드된 문서 목록을 보여줍니다."),
    Tool(name="analyze_full", func=analyze_full,
         description="특정 문서(doc_id)에 대해 품질+팩트체크 전체 분석을 수행합니다."),
]

llm = ChatOpenAI(model="gpt-5-nano")
agent = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True)

# ── Streamlit UI ──
st.set_page_config(page_title="EDU-Audit Chatbot", layout="wide")
st.title("📑 EDU-Audit 챗봇 (프로토타입)")

# 사이드바: 파일 업로드
st.sidebar.header("문서 업로드")
uploaded_file = st.sidebar.file_uploader("PDF 파일 업로드", type=["pdf"])
if uploaded_file is not None:
    with open(f"temp_{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    res = upload_document(f"temp_{uploaded_file.name}")
    st.sidebar.success(f"업로드 완료: {res}")

# 메인 채팅 영역
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("메시지를 입력하세요..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Agent 호출
    with st.chat_message("assistant"):
        response = agent.run(prompt)
        st.write(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})