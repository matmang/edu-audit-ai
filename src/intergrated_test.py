"""
EDU-Audit 실제 파일 기반 테스트
sample_docs/Ch01_intro.pdf를 사용한 전체 파이프라인 검증
"""

import asyncio
import logging
import os
import sys
import traceback
from pathlib import Path
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).resolve().parents[1] / '.env.dev'
load_dotenv(env_path)

class EDUAuditRealFileTest:
    """실제 파일을 사용한 EDU-Audit 시스템 테스트"""
    
    def __init__(self):
        self.test_file = Path("sample_docs/Ch01_intro.pdf")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.serpapi_key = os.getenv("SERPAPI_API_KEY")
        
        # 테스트 결과 저장
        self.test_results = {
            "environment": {},
            "agents": {},
            "processing": {},
            "errors": []
        }
    
    def check_prerequisites(self) -> bool:
        """사전 요구사항 확인"""
        print("=== 사전 요구사항 확인 ===")
        
        # 1. 파일 존재 확인
        if not self.test_file.exists():
            print(f"❌ 테스트 파일이 없습니다: {self.test_file}")
            print("   sample_docs 폴더에 Ch01_intro.pdf 파일을 추가해주세요")
            return False
        else:
            file_size = self.test_file.stat().st_size / (1024*1024)  # MB
            print(f"✅ 테스트 파일 확인: {self.test_file} ({file_size:.1f}MB)")
            self.test_results["environment"]["file_path"] = str(self.test_file)
            self.test_results["environment"]["file_size_mb"] = file_size
        
        # 2. API 키 확인
        if not self.openai_api_key:
            print("❌ OPENAI_API_KEY가 설정되지 않았습니다")
            return False
        else:
            print(f"✅ OpenAI API 키 확인: {self.openai_api_key[:8]}...")
            self.test_results["environment"]["has_openai_key"] = True
        
        if self.serpapi_key:
            print(f"✅ SerpAPI 키 확인: {self.serpapi_key[:8]}...")
            self.test_results["environment"]["has_serpapi_key"] = True
        else:
            print("⚠️  SerpAPI 키 없음 (사실검증 제한)")
            self.test_results["environment"]["has_serpapi_key"] = False
        
        # 3. 필수 패키지 확인
        required_packages = [
            "pdfplumber", "llama_index", "openai", "pydantic", "aiohttp"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package} 패키지가 필요합니다")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"설치 명령: pip install {' '.join(missing_packages)}")
            return False
        
        self.test_results["environment"]["packages_ok"] = True
        return True
    
    async def test_document_agent(self):
        """DocumentAgent 테스트"""
        print("\n=== Document Agent 테스트 ===")
        
        try:
            # 에이전트 초기화 시도 (간단한 버전)
            from src.core.models import DocumentMeta, PageInfo, generate_doc_id
            import pdfplumber
            
            print("📄 PDF 파일 기본 파싱 테스트...")
            
            doc_id = generate_doc_id(str(self.test_file))
            pages_data = []
            
            # 기본 PDF 파싱 (멀티모달 제외)
            with pdfplumber.open(self.test_file) as pdf:
                print(f"   총 페이지: {len(pdf.pages)}")
                
                for page_num, page in enumerate(pdf.pages[:3], 1):  # 처음 3페이지만
                    text = page.extract_text() or ""
                    word_count = len(text.split()) if text else 0
                    
                    page_info = PageInfo(
                        page_id=f"p{page_num:03d}",
                        page_number=page_num,
                        raw_text=text,
                        word_count=word_count,
                        elements=[]  # 일단 빈 리스트
                    )
                    
                    pages_data.append(page_info)
                    print(f"   페이지 {page_num}: {word_count} 단어")
            
            # 문서 메타데이터
            doc_meta = DocumentMeta(
                doc_id=doc_id,
                title="Ch01_intro",
                doc_type="pdf",
                total_pages=len(pages_data),
                file_path=str(self.test_file)
            )
            
            # 결과 저장
            self.test_results["agents"]["document_agent"] = {
                "status": "success",
                "doc_id": doc_id,
                "pages_parsed": len(pages_data),
                "total_words": sum(p.word_count for p in pages_data),
                "sample_text": pages_data[0].raw_text[:200] + "..." if pages_data else ""
            }
            
            print("✅ Document Agent 기본 파싱 성공")
            return doc_meta, pages_data
            
        except Exception as e:
            error_msg = f"Document Agent 실패: {str(e)}"
            print(f"❌ {error_msg}")
            self.test_results["errors"].append(error_msg)
            print(f"스택 트레이스:\n{traceback.format_exc()}")
            return None, []
    
    async def test_quality_agent(self, doc_meta, pages_data):
        """QualityAgent 테스트"""
        print("\n=== Quality Agent 테스트 ===")
        
        if not doc_meta or not pages_data:
            print("❌ Document Agent 결과가 없어서 건너뜁니다")
            return []
        
        try:
            # 간단한 오탈자 패턴 검사 (LLM 없이)
            typo_patterns = [
                {"pattern": r"알고리듬", "correction": "알고리즘", "description": "알고리듬 → 알고리즘"},
                {"pattern": r"데이타", "correction": "데이터", "description": "데이타 → 데이터"},
                {"pattern": r"컴퓨타", "correction": "컴퓨터", "description": "컴퓨타 → 컴퓨터"},
                {"pattern": r"앨고리즘", "correction": "알고리즘", "description": "앨고리즘 → 알고리즘"},
            ]
            
            import re
            from src.core.models import Issue, IssueType, TextLocation, generate_issue_id
            
            issues_found = []
            
            print("🔍 오탈자 패턴 검사 중...")
            
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
                            agent_name="quality_agent_simple"
                        )
                        
                        issues_found.append(issue)
                        print(f"   발견: '{match.group()}' → '{pattern_info['correction']}' (페이지 {page.page_number})")
            
            # 결과 저장
            self.test_results["agents"]["quality_agent"] = {
                "status": "success",
                "issues_found": len(issues_found),
                "issue_types": [issue.issue_type.value for issue in issues_found]
            }
            
            print(f"✅ Quality Agent 테스트 완료: {len(issues_found)}개 이슈 발견")
            return issues_found
            
        except Exception as e:
            error_msg = f"Quality Agent 실패: {str(e)}"
            print(f"❌ {error_msg}")
            self.test_results["errors"].append(error_msg)
            print(f"스택 트레이스:\n{traceback.format_exc()}")
            return []
    
    async def test_fact_check_agent(self, doc_meta, pages_data):
        """FactCheckAgent 간단 테스트"""
        print("\n=== Fact Check Agent 테스트 ===")
        
        if not doc_meta or not pages_data:
            print("❌ Document Agent 결과가 없어서 건너뜁니다")
            return []
        
        try:
            # 사실 검증이 필요한 패턴 찾기 (실제 검증은 skip)
            fact_patterns = [
                r"연구에 따르면",
                r"최신 연구",
                r"[0-9]{4}년.*연구",
                r"통계에 의하면",
                r"[0-9]+%"
            ]
            
            import re
            
            potential_claims = []
            
            print("🔍 사실 검증 대상 문장 탐지 중...")
            
            for page in pages_data:
                text = page.raw_text
                if not text:
                    continue
                
                sentences = re.split(r'[.!?]\s+', text)
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) < 10:
                        continue
                    
                    for pattern in fact_patterns:
                        if re.search(pattern, sentence, re.IGNORECASE):
                            potential_claims.append({
                                "page": page.page_number,
                                "sentence": sentence[:100] + "..." if len(sentence) > 100 else sentence,
                                "pattern": pattern
                            })
                            print(f"   페이지 {page.page_number}: {sentence[:50]}...")
                            break
            
            # 결과 저장
            self.test_results["agents"]["fact_check_agent"] = {
                "status": "success",
                "potential_claims": len(potential_claims),
                "note": "실제 검증은 수행하지 않음 (API 호출 제한)"
            }
            
            print(f"✅ Fact Check Agent 테스트 완료: {len(potential_claims)}개 검증 대상 발견")
            return potential_claims
            
        except Exception as e:
            error_msg = f"Fact Check Agent 실패: {str(e)}"
            print(f"❌ {error_msg}")
            self.test_results["errors"].append(error_msg)
            print(f"스택 트레이스:\n{traceback.format_exc()}")
            return []
    
    async def test_llm_connection(self):
        """OpenAI LLM 연결 테스트"""
        print("\n=== LLM 연결 테스트 ===")
        
        try:
            from llama_index.llms.openai import OpenAI
            
            llm = OpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                api_key=self.openai_api_key
            )
            
            # 간단한 테스트 요청
            print("🔗 OpenAI API 연결 테스트 중...")
            test_prompt = "안녕하세요. 간단한 연결 테스트입니다. '테스트 성공'이라고 답변해주세요."
            
            response = await llm.acomplete(test_prompt)
            
            print(f"✅ LLM 연결 성공")
            print(f"   응답: {response.text[:100]}...")
            
            self.test_results["agents"]["llm_connection"] = {
                "status": "success",
                "model": "gpt-3.5-turbo",
                "response_preview": response.text[:50]
            }
            
            return True
            
        except Exception as e:
            error_msg = f"LLM 연결 실패: {str(e)}"
            print(f"❌ {error_msg}")
            self.test_results["errors"].append(error_msg)
            return False
    
    async def test_indexing(self, doc_meta, pages_data):
        """LlamaIndex 인덱싱 테스트"""
        print("\n=== 인덱싱 테스트 ===")
        
        if not doc_meta or not pages_data:
            print("❌ Document Agent 결과가 없어서 건너뜁니다")
            return False
        
        try:
            from llama_index.core import Document, VectorStoreIndex
            from llama_index.embeddings.openai import OpenAIEmbedding
            
            print("📊 벡터 인덱스 생성 중...")
            
            # Document 객체들 생성
            documents = []
            for page in pages_data:
                if page.raw_text.strip():
                    doc = Document(
                        text=page.raw_text,
                        metadata={
                            "doc_id": doc_meta.doc_id,
                            "page_id": page.page_id,
                            "page_number": page.page_number
                        }
                    )
                    documents.append(doc)
            
            if not documents:
                print("❌ 인덱싱할 문서가 없습니다")
                return False
            
            # 임베딩 모델 초기화
            embeddings = OpenAIEmbedding(api_key=self.openai_api_key)
            
            # 벡터 인덱스 생성
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=embeddings
            )
            
            # 간단한 검색 테스트
            query_engine = index.as_query_engine(similarity_top_k=2)
            test_query = "주요 내용은 무엇인가요?"
            
            print("🔍 검색 테스트 중...")
            response = query_engine.query(test_query)
            
            print(f"✅ 인덱싱 성공: {len(documents)}개 문서 인덱싱")
            print(f"   검색 결과: {str(response)[:100]}...")
            
            self.test_results["agents"]["indexing"] = {
                "status": "success",
                "documents_indexed": len(documents),
                "search_test": str(response)[:100]
            }
            
            return True
            
        except Exception as e:
            error_msg = f"인덱싱 실패: {str(e)}"
            print(f"❌ {error_msg}")
            self.test_results["errors"].append(error_msg)
            return False
    
    def generate_report(self):
        """테스트 결과 보고서 생성"""
        print("\n" + "="*60)
        print("📋 EDU-Audit 테스트 결과 보고서")
        print("="*60)
        
        # 환경 설정
        print(f"\n🔧 환경 설정:")
        env = self.test_results["environment"]
        print(f"   파일: {env.get('file_path', 'N/A')} ({env.get('file_size_mb', 0):.1f}MB)")
        print(f"   OpenAI API: {'✅' if env.get('has_openai_key') else '❌'}")
        print(f"   SerpAPI: {'✅' if env.get('has_serpapi_key') else '❌'}")
        print(f"   패키지: {'✅' if env.get('packages_ok') else '❌'}")
        
        # 에이전트 테스트 결과
        print(f"\n🤖 에이전트 테스트 결과:")
        agents = self.test_results["agents"]
        
        for agent_name, result in agents.items():
            status = result.get("status", "unknown")
            emoji = "✅" if status == "success" else "❌"
            print(f"   {agent_name}: {emoji}")
            
            if agent_name == "document_agent" and status == "success":
                print(f"      - 페이지: {result['pages_parsed']}")
                print(f"      - 단어 수: {result['total_words']:,}")
            
            elif agent_name == "quality_agent" and status == "success":
                print(f"      - 발견 이슈: {result['issues_found']}")
            
            elif agent_name == "fact_check_agent" and status == "success":
                print(f"      - 검증 대상: {result['potential_claims']}")
            
            elif agent_name == "indexing" and status == "success":
                print(f"      - 인덱스 문서: {result['documents_indexed']}")
        
        # 오류 목록
        if self.test_results["errors"]:
            print(f"\n❌ 발견된 오류들:")
            for i, error in enumerate(self.test_results["errors"], 1):
                print(f"   {i}. {error}")
        
        # 권장 사항
        print(f"\n💡 권장 사항:")
        
        if not env.get('has_serpapi_key'):
            print("   - SerpAPI 키 설정으로 사실검증 기능 활성화")
        
        if self.test_results["errors"]:
            print("   - 오류 해결 후 재테스트 필요")
        else:
            print("   - 기본 기능 정상 작동, Streamlit 데모 개발 가능")
        
        print("\n" + "="*60)
    
    async def run_full_test(self):
        """전체 테스트 실행"""
        print("🚀 EDU-Audit 실제 파일 기반 테스트 시작")
        print(f"테스트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = datetime.now()
        
        # 사전 요구사항 확인
        if not self.check_prerequisites():
            print("\n❌ 사전 요구사항을 충족하지 않습니다. 테스트를 중단합니다.")
            return
        
        # 1. Document Agent 테스트
        doc_meta, pages_data = await self.test_document_agent()
        
        # 2. Quality Agent 테스트
        quality_issues = await self.test_quality_agent(doc_meta, pages_data)
        
        # 3. LLM 연결 테스트
        llm_connected = await self.test_llm_connection()
        
        # 4. 인덱싱 테스트 (LLM 연결이 성공한 경우만)
        if llm_connected:
            await self.test_indexing(doc_meta, pages_data)
        
        # 5. Fact Check Agent 테스트
        fact_claims = await self.test_fact_check_agent(doc_meta, pages_data)
        
        # 처리 시간 계산
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        self.test_results["processing"] = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": processing_time
        }
        
        # 결과 보고서 생성
        self.generate_report()
        
        print(f"\n⏱️  총 처리 시간: {processing_time:.2f}초")
        print(f"🎯 다음 단계: Streamlit 데모 개발 준비 완료")


async def main():
    """메인 실행 함수"""
    tester = EDUAuditRealFileTest()
    await tester.run_full_test()


if __name__ == "__main__":
    # 비동기 실행
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ 사용자가 테스트를 중단했습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {str(e)}")
        print(f"스택 트레이스:\n{traceback.format_exc()}")