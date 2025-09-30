"""
EDU-Audit Streamlit Chat Interface (개선 버전)
LangChain Agent + Streamlit으로 구현한 교육자료 품질 검수 챗봇
"""

import streamlit as st
import requests
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime

# LangChain imports (최신 버전)
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from dotenv import load_dotenv
import os

# 환경변수 로드
env_path = Path(__file__).resolve().parents[3] / '.env.dev'
load_dotenv(env_path)

# 설정
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Tools import (위에서 만든 tools 모듈)
try:
    from tools import create_edu_audit_tools
    TOOLS_AVAILABLE = True
except ImportError as e:
    print(e)
    TOOLS_AVAILABLE = False
    st.error("⚠️ tools 모듈을 찾을 수 없습니다. chatbot 디렉토리를 확인하세요.")


# ═══════════════════════════════════════════════════════
# 페이지 설정
# ═══════════════════════════════════════════════════════

st.set_page_config(
    page_title="EDU-Audit 챗봇",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .doc-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stChatMessage {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# 헬퍼 함수들
# ═══════════════════════════════════════════════════════

def check_server_health() -> bool:
    """서버 상태 확인"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def upload_document_to_server(file) -> Dict[str, Any]:
    """파일을 서버에 업로드"""
    try:
        # 임시 파일로 저장
        temp_path = Path(f"temp_{file.name}")
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        
        # 서버에 업로드
        with open(temp_path, "rb") as f:
            response = requests.post(
                f"{BASE_URL}/document/upload",
                files={"file": f}
            )
        
        # 임시 파일 삭제
        temp_path.unlink()
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text}
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_documents_list() -> List[Dict]:
    """문서 목록 가져오기"""
    try:
        response = requests.get(f"{BASE_URL}/document/list")
        if response.status_code == 200:
            data = response.json()
            return data.get("documents", [])
        return []
    except:
        return []


def format_doc_info(doc: Dict) -> str:
    """문서 정보를 포맷팅"""
    return f"""
    **📄 {doc.get('title', 'Unknown')}**
    - ID: `{doc.get('doc_id', 'N/A')}`
    - 페이지: {doc.get('total_pages', 0)}
    - 업로드: {doc.get('created_at', 'N/A')}
    """


# ═══════════════════════════════════════════════════════
# Agent 초기화
# ═══════════════════════════════════════════════════════

@st.cache_resource
def initialize_agent():
    """LangChain Agent 초기화 (캐싱)"""
    
    if not TOOLS_AVAILABLE:
        return None
    
    # Tools 생성
    tools = create_edu_audit_tools(BASE_URL)
    
    # LLM 설정
    llm = ChatOpenAI(
        model="gpt-5-nano",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )
    
    # System Prompt
    system_prompt = """당신은 EDU-Audit 교육자료 품질 검수 전문 AI 어시스턴트입니다.

당신의 역할:
- 사용자가 업로드한 교육 문서의 품질을 검수합니다
- 오탈자, 문법, 가독성 문제를 찾아냅니다
- 팩트체킹을 통해 정보의 정확성을 검증합니다
- 문서 내 특정 내용을 검색해줍니다

사용 가능한 도구들:
- list_documents: 업로드된 문서 목록 조회
- get_document_info: 특정 문서의 상세 정보 조회
- analyze_quality: 문서 품질 분석 (오탈자, 문법 등)
- analyze_factcheck: 팩트체킹 분석
- analyze_full: 전체 분석 (품질 + 팩트체킹)
- search_document: 문서 내 검색

대화 원칙:
1. 사용자가 문서를 업로드하면, 먼저 list_documents로 확인하세요
2. 분석 요청 시 적절한 도구를 선택하세요
3. 결과를 명확하고 친절하게 설명하세요
4. 한국어로 자연스럽게 대화하세요
5. doc_id가 필요한 경우, 사용자에게 물어보거나 최근 문서를 자동으로 선택하세요

항상 사용자의 의도를 이해하고 최선의 도움을 제공하세요!
"""
    
    # Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Agent 생성
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Agent Executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        return_intermediate_steps=False
    )
    
    return agent_executor


# ═══════════════════════════════════════════════════════
# 세션 상태 초기화
# ═══════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

if "current_doc_id" not in st.session_state:
    st.session_state.current_doc_id = None


# ═══════════════════════════════════════════════════════
# 사이드바: 파일 업로드 & 문서 관리
# ═══════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📁 문서 관리")
    
    # 서버 상태 체크
    server_status = check_server_health()
    if server_status:
        st.success("✅ 서버 연결됨")
    else:
        st.error("❌ 서버 연결 실패")
        st.info(f"서버 주소: {BASE_URL}")
    
    st.markdown("---")
    
    # 파일 업로드
    st.markdown("### 📤 파일 업로드")
    uploaded_file = st.file_uploader(
        "PDF 파일을 업로드하세요",
        type=["pdf"],
        help="교육 자료 PDF 파일만 지원됩니다"
    )
    
    if uploaded_file is not None:
        if st.button("업로드 시작", type="primary", use_container_width=True):
            with st.spinner("업로드 중..."):
                result = upload_document_to_server(uploaded_file)
                
                if result["success"]:
                    doc_data = result["data"]
                    doc_meta = doc_data.get("doc_meta", {})
                    
                    st.success(f"✅ 업로드 완료!")
                    st.info(f"**문서 ID:** `{doc_meta.get('doc_id', 'N/A')}`")
                    
                    # 현재 문서 ID 저장
                    st.session_state.current_doc_id = doc_meta.get('doc_id')
                    
                    # 문서 목록 새로고침
                    st.session_state.uploaded_docs = get_documents_list()
                    
                    # 챗봇에 자동 메시지 추가
                    auto_message = f"'{uploaded_file.name}' 파일이 업로드되었습니다. 문서 ID는 `{doc_meta.get('doc_id')}`입니다."
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": auto_message
                    })
                    
                    st.rerun()
                else:
                    st.error(f"❌ 업로드 실패: {result['error']}")
    
    st.markdown("---")
    
    # 업로드된 문서 목록
    st.markdown("### 📚 업로드된 문서")
    
    if st.button("목록 새로고침", use_container_width=True):
        st.session_state.uploaded_docs = get_documents_list()
        st.rerun()
    
    docs = st.session_state.uploaded_docs or get_documents_list()
    
    if docs:
        for doc in docs:
            with st.expander(f"📄 {doc.get('title', 'Unknown')[:30]}..."):
                st.markdown(format_doc_info(doc))
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("선택", key=f"select_{doc['doc_id']}", use_container_width=True):
                        st.session_state.current_doc_id = doc['doc_id']
                        st.success(f"선택됨: {doc['doc_id'][:8]}...")
                
                with col2:
                    if st.button("분석", key=f"analyze_{doc['doc_id']}", use_container_width=True):
                        st.session_state.current_doc_id = doc['doc_id']
                        # 채팅에 자동 명령 추가
                        st.session_state.messages.append({
                            "role": "user",
                            "content": f"{doc['doc_id']} 문서를 전체 분석해줘"
                        })
                        st.rerun()
    else:
        st.info("업로드된 문서가 없습니다")
    
    st.markdown("---")
    
    # 현재 선택된 문서
    if st.session_state.current_doc_id:
        st.markdown("### 🎯 현재 선택")
        st.code(st.session_state.current_doc_id)
        
        if st.button("선택 해제", use_container_width=True):
            st.session_state.current_doc_id = None
            st.rerun()
    
    st.markdown("---")
    
    # 대화 초기화
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()


# ═══════════════════════════════════════════════════════
# 메인 영역: 챗봇 인터페이스
# ═══════════════════════════════════════════════════════

st.markdown('<div class="main-header">📚 EDU-Audit 챗봇</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">교육 자료 품질 검수 AI 어시스턴트</div>', unsafe_allow_html=True)

# 빠른 명령어 버튼
st.markdown("### 🚀 빠른 명령")
col1, col2, col3, col4 = st.columns(4)

# 버튼 클릭 시 실행할 명령 저장
quick_command = None

with col1:
    if st.button("📋 문서 목록", use_container_width=True):
        quick_command = "업로드된 문서 목록을 보여줘"

with col2:
    if st.button("✅ 품질 검사", use_container_width=True):
        if st.session_state.current_doc_id:
            quick_command = f"{st.session_state.current_doc_id} 문서의 품질을 분석해줘"
        else:
            st.warning("먼저 문서를 선택하세요")

with col3:
    if st.button("🔍 팩트체크", use_container_width=True):
        if st.session_state.current_doc_id:
            quick_command = f"{st.session_state.current_doc_id} 문서를 팩트체크해줘"
        else:
            st.warning("먼저 문서를 선택하세요")

with col4:
    if st.button("📊 전체 분석", use_container_width=True):
        if st.session_state.current_doc_id:
            quick_command = f"{st.session_state.current_doc_id} 문서를 전체 분석해줘"
        else:
            st.warning("먼저 문서를 선택하세요")

st.markdown("---")

# 채팅 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 빠른 명령 처리 (버튼 클릭 시)
if quick_command:
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": quick_command})
    
    with st.chat_message("user"):
        st.markdown(quick_command)
    
    # Agent 응답
    with st.chat_message("assistant"):
        with st.spinner("생각하는 중..."):
            try:
                agent_executor = initialize_agent()
                
                if agent_executor is None:
                    response = "❌ Agent를 초기화할 수 없습니다. tools 모듈을 확인하세요."
                else:
                    # chat_history 준비
                    chat_history = []
                    for msg in st.session_state.messages[:-1]:
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        else:
                            chat_history.append(AIMessage(content=msg["content"]))
                    
                    # Agent 실행
                    result = agent_executor.invoke({
                        "input": quick_command,
                        "chat_history": chat_history
                    })
                    
                    response = result.get("output", "응답을 생성할 수 없습니다.")
                
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                error_msg = f"❌ 오류 발생: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# 채팅 입력
if prompt := st.chat_input("메시지를 입력하세요... (예: '문서 목록 보여줘', '품질 검사해줘')"):
    
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Agent 응답
    with st.chat_message("assistant"):
        with st.spinner("생각하는 중..."):
            try:
                # Agent 실행
                agent_executor = initialize_agent()
                
                if agent_executor is None:
                    response = "❌ Agent를 초기화할 수 없습니다. tools 모듈을 확인하세요."
                else:
                    # chat_history 준비
                    chat_history = []
                    for msg in st.session_state.messages[:-1]:  # 마지막 사용자 메시지 제외
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        else:
                            chat_history.append(AIMessage(content=msg["content"]))
                    
                    # Agent 실행
                    result = agent_executor.invoke({
                        "input": prompt,
                        "chat_history": chat_history
                    })
                    
                    response = result.get("output", "응답을 생성할 수 없습니다.")
                
                st.markdown(response)
                
                # 응답 저장
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                error_msg = f"❌ 오류 발생: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })


# ═══════════════════════════════════════════════════════
# 푸터
# ═══════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>💡 <b>사용 팁:</b> 자연어로 명령하세요! "문서 분석해줘", "오탈자 찾아줘" 등</p>
    <p>Powered by LangChain + OpenAI | EDU-Audit v0.1.0</p>
</div>
""", unsafe_allow_html=True)