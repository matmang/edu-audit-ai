"""
EDU-Audit Streamlit 챗봇 데모
교수/연구자를 위한 대화형 교육 콘텐츠 검수 시스템
"""

import streamlit as st
import asyncio
import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# 환경 설정
from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).resolve().parents[1] / '.env.dev'
load_dotenv(env_path)

# 우리 시스템 import
import sys
sys.path.append('.')

from src.core.models import DocumentMeta, PageInfo, Issue, IssueType

# 페이지 설정
st.set_page_config(
    page_title="EDU-Audit: 교육 콘텐츠 검수 시스템",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_document" not in st.session_state:
    st.session_state.current_document = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "processing" not in st.session_state:
    st.session_state.processing = False

class EDUAuditChatbot:
    """EDU-Audit 챗봇 클래스"""
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            st.error("OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
            st.stop()
    
    async def analyze_document(self, file_path: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """문서 분석 실행"""
        try:
            # 간단한 분석 (테스트에서 검증된 기능만 사용)
            from src.core.models import DocumentMeta, PageInfo, Issue, IssueType, TextLocation, generate_doc_id, generate_issue_id
            import pdfplumber
            import re
            
            # 1. PDF 파싱
            doc_id = generate_doc_id(file_path)
            pages_data = []
            
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                
                # 진행률 표시를 위한 placeholder
                progress_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # 진행률 업데이트
                    progress = page_num / total_pages
                    progress_bar.progress(progress)
                    progress_placeholder.text(f"페이지 {page_num}/{total_pages} 파싱 중...")
                    
                    text = page.extract_text() or ""
                    word_count = len(text.split()) if text else 0
                    
                    page_info = PageInfo(
                        page_id=f"p{page_num:03d}",
                        page_number=page_num,
                        raw_text=text,
                        word_count=word_count,
                        elements=[]
                    )
                    pages_data.append(page_info)
                    
                    # 페이지가 많은 경우 속도 조절
                    if page_num % 5 == 0:
                        await asyncio.sleep(0.1)
                
                progress_placeholder.empty()
                progress_bar.empty()
            
            # 문서 메타데이터
            doc_meta = DocumentMeta(
                doc_id=doc_id,
                title=Path(file_path).stem,
                doc_type="pdf",
                total_pages=len(pages_data),
                file_path=file_path
            )
            
            # 2. 품질 검사 (오탈자 패턴)
            st.write("🔍 품질 검사 진행 중...")
            issues_found = await self._perform_quality_check(doc_meta, pages_data)
            
            # 3. 간단한 통계 생성
            total_words = sum(page.word_count for page in pages_data)
            
            results = {
                "document": {
                    "title": doc_meta.title,
                    "pages": len(pages_data),
                    "words": total_words,
                    "file_size": Path(file_path).stat().st_size / 1024 / 1024  # MB
                },
                "quality_issues": issues_found,
                "summary": {
                    "total_issues": len(issues_found),
                    "typo_issues": len([i for i in issues_found if i.issue_type == IssueType.TYPO]),
                    "pages_with_issues": len(set(i.page_id for i in issues_found))
                },
                "pages_data": pages_data,
                "doc_meta": doc_meta
            }
            
            return results
            
        except Exception as e:
            st.error(f"문서 분석 중 오류 발생: {str(e)}")
            return {}
    
    async def _perform_quality_check(self, doc_meta: DocumentMeta, pages_data: List[PageInfo]) -> List[Issue]:
        """품질 검사 수행"""
        # 테스트에서 검증된 오탈자 패턴들
        typo_patterns = [
            {"pattern": r"알고리듬", "correction": "알고리즘", "description": "알고리듬 → 알고리즘"},
            {"pattern": r"데이타", "correction": "데이터", "description": "데이타 → 데이터"},
            {"pattern": r"컴퓨타", "correction": "컴퓨터", "description": "컴퓨타 → 컴퓨터"},
            {"pattern": r"앨고리즘", "correction": "알고리즘", "description": "앨고리즘 → 알고리즘"},
            {"pattern": r"머신러닝", "correction": "머신러닝", "description": "기계학습과 혼용 확인"},
            {"pattern": r"딥러닝", "correction": "딥러닝", "description": "심층학습과 혼용 확인"},
        ]
        
        import re
        from src.core.models import TextLocation, generate_issue_id
        
        issues_found = []
        
        for page in pages_data:
            text = page.raw_text
            if not text:
                continue
            
            for pattern_info in typo_patterns:
                pattern = pattern_info["pattern"]
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    location = TextLocation(start=match.start(), end=match.end())
                    
                    issue_id = generate_issue_id(
                        doc_meta.doc_id, page.page_id, location, IssueType.TYPO
                    )
                    
                    issue = Issue(
                        issue_id=issue_id,
                        doc_id=doc_meta.doc_id,
                        page_id=page.page_id,
                        issue_type=IssueType.TYPO,
                        text_location=location,
                        original_text=match.group(),
                        message=pattern_info["description"],
                        suggestion=pattern_info["correction"],
                        confidence=0.95,
                        confidence_level="high",
                        agent_name="quality_agent"
                    )
                    
                    issues_found.append(issue)
        
        return issues_found
    
    def format_chat_response(self, response_type: str, data: Any) -> str:
        """챗봇 응답 포맷팅"""
        if response_type == "analysis_complete":
            doc_info = data["document"]
            summary = data["summary"]
            
            response = f"""
**📋 문서 분석 완료!**

**📄 문서 정보:**
- 제목: {doc_info['title']}
- 페이지 수: {doc_info['pages']}개
- 총 단어 수: {doc_info['words']:,}개
- 파일 크기: {doc_info['file_size']:.1f}MB

**🔍 품질 검사 결과:**
- 발견된 이슈: **{summary['total_issues']}개**
- 오탈자 관련: {summary['typo_issues']}개
- 이슈가 있는 페이지: {summary['pages_with_issues']}개

추가로 궁금한 내용이 있으시면 언제든 질문해주세요!
예: "3페이지의 오탈자를 자세히 보여줘", "가장 많은 오류가 있는 페이지는 어디야?"
"""
            return response
        
        elif response_type == "issue_details":
            issues = data
            if not issues:
                return "해당 조건에 맞는 이슈가 발견되지 않았습니다."
            
            response = f"**발견된 이슈 상세 ({len(issues)}개):**\n\n"
            for i, issue in enumerate(issues, 1):
                response += f"{i}. **{issue.original_text}**\n"
                response += f"   - 위치: {issue.page_id}\n"
                response += f"   - 문제: {issue.message}\n"
                response += f"   - 제안: {issue.suggestion}\n"
                response += f"   - 신뢰도: {issue.confidence:.2f}\n\n"
            
            return response
        
        elif response_type == "page_analysis":
            page_num, issues, content_preview = data
            response = f"**📄 페이지 {page_num} 분석 결과:**\n\n"
            
            if issues:
                response += f"**발견된 이슈 ({len(issues)}개):**\n"
                for issue in issues:
                    response += f"- **{issue.original_text}**: {issue.message}\n"
                response += "\n"
            else:
                response += "이 페이지에서는 이슈가 발견되지 않았습니다.\n\n"
            
            response += f"**내용 미리보기:**\n{content_preview[:200]}..."
            return response
        
        else:
            return "죄송합니다. 요청을 처리할 수 없습니다."

# 챗봇 인스턴스 생성
@st.cache_resource
def get_chatbot():
    return EDUAuditChatbot()

chatbot = get_chatbot()

def generate_simple_response(prompt: str, analysis_results: Dict) -> str:
    """간단한 키워드 기반 응답 생성"""
    prompt_lower = prompt.lower()
    
    # 페이지별 질문
    if "페이지" in prompt and any(char.isdigit() for char in prompt):
        # 페이지 번호 추출
        import re
        page_nums = re.findall(r'\d+', prompt)
        if page_nums:
            page_num = int(page_nums[0])
            
            # 해당 페이지 이슈 찾기
            page_issues = [
                issue for issue in analysis_results["quality_issues"]
                if issue.page_number == page_num
            ]
            
            # 해당 페이지 내용 찾기
            page_content = ""
            for page_data in analysis_results["pages_data"]:
                if page_data.page_number == page_num:
                    page_content = page_data.raw_text[:200]
                    break
            
            return chatbot.format_chat_response("page_analysis", (page_num, page_issues, page_content))
    
    # 전체 이슈 요약
    elif "요약" in prompt or "전체" in prompt:
        return chatbot.format_chat_response("issue_details", analysis_results["quality_issues"][:5])
    
    # 오탈자 관련 질문
    elif "오탈자" in prompt or "오류" in prompt:
        typo_issues = [
            issue for issue in analysis_results["quality_issues"]
            if issue.issue_type == IssueType.TYPO
        ]
        return chatbot.format_chat_response("issue_details", typo_issues)
    
    # 통계 질문
    elif "통계" in prompt or "얼마나" in prompt:
        summary = analysis_results["summary"]
        return f"""
**📊 문서 통계:**
- 총 페이지: {analysis_results['document']['pages']}개
- 총 단어: {analysis_results['document']['words']:,}개
- 발견된 이슈: {summary['total_issues']}개
- 이슈가 있는 페이지: {summary['pages_with_issues']}개

평균적으로 페이지당 {summary['total_issues'] / analysis_results['document']['pages']:.1f}개의 이슈가 발견되었습니다.
"""
    
    # 기본 응답
    else:
        return """
다음과 같은 질문을 해보세요:

- "3페이지의 오탈자를 보여줘"
- "전체 이슈를 요약해줘"  
- "통계를 알려줘"
- "오탈자가 몇 개야?"

더 구체적인 질문을 주시면 정확한 답변을 드릴 수 있습니다.
"""

# 메인 UI
st.title("📚 EDU-Audit: 교육 콘텐츠 검수 시스템")
st.markdown("**AI 기반 교육 자료 품질 검사 및 사실 검증 도구**")

# 사이드바
with st.sidebar:
    st.header("📁 문서 업로드")
    
    uploaded_file = st.file_uploader(
        "PDF 파일을 업로드하세요",
        type=["pdf"],
        help="교육 자료 PDF 파일을 선택해주세요"
    )
    
    if uploaded_file is not None:
        # 임시 파일 저장
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        temp_file_path = temp_dir / uploaded_file.name
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        st.success(f"파일 업로드 완료: {uploaded_file.name}")
        
        # 분석 옵션
        st.header("⚙️ 분석 설정")
        
        analysis_type = st.selectbox(
            "분석 수준",
            ["기본 (오탈자 위주)", "표준 (품질검사 포함)", "종합 (사실검증 포함)"],
            index=1
        )
        
        # 분석 시작 버튼
        if st.button("🔍 분석 시작", disabled=st.session_state.processing):
            if not st.session_state.processing:
                st.session_state.processing = True
                
                with st.spinner("문서 분석 중입니다. 잠시만 기다려주세요..."):
                    # 비동기 분석 실행
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    results = loop.run_until_complete(
                        chatbot.analyze_document(str(temp_file_path), analysis_type)
                    )
                    
                    if results:
                        st.session_state.analysis_results = results
                        st.session_state.current_document = temp_file_path
                        
                        # 분석 완료 메시지 추가
                        response = chatbot.format_chat_response("analysis_complete", results)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.now()
                        })
                        
                        st.success("분석 완료!")
                    
                    loop.close()
                
                st.session_state.processing = False
                st.rerun()
    
    # 현재 문서 정보
    if st.session_state.analysis_results:
        st.header("📊 현재 문서")
        doc_info = st.session_state.analysis_results["document"]
        
        st.metric("페이지 수", doc_info["pages"])
        st.metric("총 단어 수", f"{doc_info['words']:,}")
        st.metric("발견된 이슈", st.session_state.analysis_results["summary"]["total_issues"])

# 메인 챗 인터페이스
st.header("💬 AI 어시스턴트와 대화")

# 사용 예시 안내
if not st.session_state.messages and not st.session_state.analysis_results:
    st.info("""
    **사용 방법:**
    1. 왼쪽에서 PDF 파일을 업로드하세요
    2. 분석 완료 후 다음과 같은 질문을 해보세요:
    
    - "3페이지의 오탈자를 보여줘"
    - "가장 많은 오류가 있는 페이지는?"
    - "전체 이슈를 요약해줘"
    - "특정 단어의 일관성을 확인해줘"
    """)

# 채팅 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # 타임스탬프 표시
        st.caption(f"⏰ {message['timestamp'].strftime('%H:%M:%S')}")

# 채팅 입력
if prompt := st.chat_input("궁금한 내용을 질문해보세요..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "timestamp": datetime.now()
    })
    
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # AI 응답 생성
    with st.chat_message("assistant"):
        if not st.session_state.analysis_results:
            response = "먼저 PDF 파일을 업로드하고 분석을 실행해주세요."
        else:
            # 간단한 키워드 기반 응답 (실제 LLM 대신)
            response = generate_simple_response(prompt, st.session_state.analysis_results)
        
        st.markdown(response)
        
        # AI 응답 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now()
        })

# 푸터
st.markdown("---")
st.markdown("**EDU-Audit v1.0** | 교육 콘텐츠 품질 검수 시스템")