"""
EDU-Audit Fact Check Agent - Efficient & Selective
선택적 팩트체킹 에이전트 (LLM 필터 → 검색 → 대조 → 이슈화)
"""

import asyncio
import logging
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass

import aiohttp
from openai import AsyncOpenAI

from src.core.models import (
    DocumentMeta, Issue, IssueType, TextLocation,
    generate_issue_id
)

logger = logging.getLogger(__name__)

@dataclass
class FactCheckTrigger:
    """팩트체크 필요 여부 판단 결과"""
    factcheck_required: bool
    reason: str
    keywords: List[str]
    confidence: float

@dataclass
class SearchResult:
    """외부 검색 결과"""
    title: str
    url: str
    snippet: str
    source_domain: str

@dataclass
class FactVerification:
    """팩트체크 검증 결과"""
    claim: str
    is_accurate: bool
    is_outdated: bool
    confidence: float
    reasoning: str
    search_results: List[SearchResult]

class FactCheckAgent:
    """
    선택적 팩트체킹 에이전트
    
    파이프라인:
    1. LLM 필터: 팩트체크 필요 여부 판단
    2. 검색 실행: 필요한 경우만 외부 검색
    3. 대조 단계: 검색 결과 vs 슬라이드 내용 비교
    4. 이슈화: 문제가 있는 경우만 Issue 생성
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        serpapi_key: Optional[str] = None,
        model: str = "gpt-5-nano",
        max_search_results: int = 5,
        search_timeout: int = 10
    ):
        self.openai_api_key = openai_api_key
        self.serpapi_key = serpapi_key
        self.model = model
        self.max_search_results = max_search_results
        self.search_timeout = search_timeout
        
        if not openai_api_key:
            raise ValueError("OpenAI API Key가 필요합니다.")
        
        # OpenAI 클라이언트
        self.client = AsyncOpenAI(api_key=openai_api_key)
        
        # HTTP 세션 (검색용)
        self.session = None
        
        # 캐시 (중복 검색 방지)
        self.search_cache: Dict[str, List[SearchResult]] = {}
        self.verification_cache: Dict[str, FactVerification] = {}
        
        logger.info(f"FactCheckAgent 초기화 완료")
        logger.info(f"  모델: {model}")
        logger.info(f"  검색 API: {'활성화' if serpapi_key else '비활성화'}")
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.search_timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    async def analyze_document(self, document_agent, doc_id: str) -> List[Issue]:
        """
        DocumentAgent에서 처리된 문서의 팩트체킹 분석
        
        Args:
            document_agent: DocumentAgent 인스턴스
            doc_id: 문서 ID
            
        Returns:
            List[Issue]: 발견된 팩트체킹 이슈들
        """
        logger.info(f"팩트체킹 분석 시작: {doc_id}")
        
        # DocumentAgent에서 슬라이드 데이터 가져오기
        doc_meta = document_agent.get_document(doc_id)
        slide_data_list = document_agent.get_slide_data(doc_id)
        
        if not doc_meta or not slide_data_list:
            logger.warning(f"문서 데이터가 없습니다: {doc_id}")
            return []
        
        all_issues = []
        
        async with self:
            # 1단계: 각 슬라이드별 팩트체크 필요 여부 판단
            factcheck_candidates = []
            
            for slide_data in slide_data_list:
                try:
                    trigger = await self._check_factcheck_trigger(slide_data)
                    
                    if trigger.factcheck_required:
                        factcheck_candidates.append({
                            "slide_data": slide_data,
                            "trigger": trigger
                        })
                        logger.info(f"팩트체크 대상: {slide_data['page_id']} - {trigger.reason}")
                    
                    # API 레이트 제한
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"트리거 판단 실패 {slide_data['page_id']}: {str(e)}")
                    continue
            
            logger.info(f"팩트체크 대상: {len(factcheck_candidates)}/{len(slide_data_list)} 슬라이드")
            
            # 2단계: 팩트체크 필요한 슬라이드만 처리
            for candidate in factcheck_candidates:
                try:
                    slide_issues = await self._verify_slide_facts(
                        candidate["slide_data"], 
                        candidate["trigger"],
                        doc_meta
                    )
                    all_issues.extend(slide_issues)
                    
                    # 검색 API 레이트 제한
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"슬라이드 팩트체크 실패 {candidate['slide_data']['page_id']}: {str(e)}")
                    continue
        
        logger.info(f"팩트체킹 분석 완료: {len(all_issues)}개 이슈 발견")
        return all_issues
    
    async def _check_factcheck_trigger(self, slide_data: Dict[str, Any]) -> FactCheckTrigger:
        """1단계: LLM으로 팩트체크 필요 여부 판단"""
        
        # 분석할 텍스트 준비
        analysis_text = ""
        if slide_data.get("caption"):
            analysis_text += f"[캡션] {slide_data['caption']}"
        
        if slide_data.get("slide_text"):
            analysis_text += f"\n[텍스트] {slide_data['slide_text']}"
        
        if not analysis_text.strip():
            return FactCheckTrigger(
                factcheck_required=False,
                reason="분석할 텍스트 없음",
                keywords=[],
                confidence=1.0
            )
        
        trigger_prompt = f"""다음 교육 슬라이드 내용을 보고, 외부 검색을 통한 사실 확인이 필요한지 판단해주세요.

슬라이드 내용:
{analysis_text}

**팩트체크가 필요한 경우:**
- 구체적인 수치, 통계, 데이터 (예: "사용자 수 1억명", "정확도 95%")
- 최신 연구 결과, 발표 내용 (예: "2024년 연구", "최근 발표")
- 회사/제품 정보, 출시일 (예: "GPT-4 출시", "새로운 기능")
- 법규, 정책, 현황 (예: "현재 규제", "정부 정책")
- 시의성 있는 사건, 동향 (예: "올해 트렌드", "최근 변화")

**팩트체크가 불필요한 경우:**
- 수학 공식, 알고리즘 설명 (예: "θ = θ - η∇J(θ)")
- 기본 개념 정의 (예: "머신러닝이란", "분류와 회귀")
- 일반적인 설명, 원리 (예: "신경망 구조", "학습 과정")
- 예시, 비유, 교육적 설명

JSON 형식으로 응답해주세요:
{{
    "factcheck_required": true|false,
    "reason": "판단 이유 (한 문장)",
    "keywords": ["검색할", "핵심", "키워드"],
    "confidence": 0.0-1.0
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": trigger_prompt}
                ],
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # JSON 파싱
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            
            return FactCheckTrigger(
                factcheck_required=result.get("factcheck_required", False),
                reason=result.get("reason", "이유 없음"),
                keywords=result.get("keywords", []),
                confidence=result.get("confidence", 0.5)
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"트리거 판단 JSON 파싱 실패: {response_text[:100]}... - {str(e)}")
            return FactCheckTrigger(
                factcheck_required=False,
                reason="응답 파싱 실패",
                keywords=[],
                confidence=0.1
            )
        except Exception as e:
            logger.error(f"트리거 판단 실패: {str(e)}")
            return FactCheckTrigger(
                factcheck_required=False,
                reason="판단 과정 오류",
                keywords=[],
                confidence=0.1
            )
    
    async def _verify_slide_facts(
        self, 
        slide_data: Dict[str, Any], 
        trigger: FactCheckTrigger,
        doc_meta: DocumentMeta
    ) -> List[Issue]:
        """2-4단계: 검색 → 대조 → 이슈화"""
        
        issues = []
        
        # 검색 키워드 준비
        search_queries = self._prepare_search_queries(slide_data, trigger)
        
        for query in search_queries:
            try:
                # 캐시 확인
                cache_key = f"{query}:{slide_data['page_id']}"
                if cache_key in self.verification_cache:
                    verification = self.verification_cache[cache_key]
                    logger.info(f"캐시에서 검증 결과 사용: {query}")
                else:
                    # 2단계: 외부 검색
                    search_results = await self._search_external(query)
                    
                    # 3단계: 검색 결과와 슬라이드 내용 대조
                    verification = await self._compare_with_search_results(
                        slide_data, query, search_results
                    )
                    
                    # 캐시에 저장
                    self.verification_cache[cache_key] = verification
                
                # 4단계: 문제가 있는 경우 이슈 생성
                if not verification.is_accurate or verification.is_outdated:
                    issue = self._create_fact_issue(
                        slide_data, verification, doc_meta
                    )
                    issues.append(issue)
                
            except Exception as e:
                logger.warning(f"팩트체크 실패 '{query}': {str(e)}")
                continue
        
        return issues
    
    def _prepare_search_queries(self, slide_data: Dict[str, Any], trigger: FactCheckTrigger) -> List[str]:
        """검색 쿼리 준비"""
        queries = []
        
        # 트리거에서 추출한 키워드 사용
        if trigger.keywords:
            # 키워드 조합으로 검색 쿼리 생성
            main_keywords = " ".join(trigger.keywords[:3])  # 상위 3개만
            queries.append(main_keywords)
        
        # 캡션에서 숫자나 구체적 정보 추출
        text_content = ""
        if slide_data.get("caption"):
            text_content += slide_data["caption"]
        if slide_data.get("slide_text"):
            text_content += " " + slide_data["slide_text"]
        
        # 숫자가 포함된 문장 추출
        sentences = re.split(r'[.!?]\s+', text_content)
        for sentence in sentences:
            if re.search(r'\d', sentence) and len(sentence) > 10:
                # 문장에서 핵심 부분만 추출 (처음 50자)
                clean_sentence = sentence.strip()[:50]
                if clean_sentence not in [q[:50] for q in queries]:
                    queries.append(clean_sentence)
        
        return queries[:2]  # 최대 2개 쿼리만
    
    async def _search_external(self, query: str) -> List[SearchResult]:
        """2단계: 외부 검색 실행"""
        
        # 캐시 확인
        if query in self.search_cache:
            logger.info(f"검색 캐시 사용: {query}")
            return self.search_cache[query]
        
        search_results = []
        
        try:
            if self.serpapi_key and self.session:
                search_results = await self._search_with_serpapi(query)
            else:
                # 검색 API가 없는 경우 더미 결과
                logger.warning("검색 API 키가 없어 더미 결과 사용")
                search_results = self._generate_dummy_results(query)
            
            # 캐시에 저장
            self.search_cache[query] = search_results
            
        except Exception as e:
            logger.warning(f"외부 검색 실패 '{query}': {str(e)}")
            search_results = self._generate_dummy_results(query)
        
        return search_results
    
    async def _search_with_serpapi(self, query: str) -> List[SearchResult]:
        """SerpAPI를 통한 Google 검색"""
        
        try:
            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "api_key": self.serpapi_key,
                "engine": "google",
                "num": self.max_search_results,
                "gl": "kr",  # 한국 결과
                "hl": "ko"   # 한국어
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"SerpAPI 응답 오류: {response.status}")
                    return []
                
                data = await response.json()
                results = []
                
                for item in data.get("organic_results", []):
                    result = SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source_domain=self._extract_domain(item.get("link", ""))
                    )
                    results.append(result)
                
                logger.info(f"SerpAPI 검색 완료: {len(results)}개 결과")
                return results
                
        except Exception as e:
            logger.warning(f"SerpAPI 검색 실패: {str(e)}")
            return []
    
    def _generate_dummy_results(self, query: str) -> List[SearchResult]:
        """더미 검색 결과 (테스트/데모용)"""
        return [
            SearchResult(
                title=f"Search result for: {query}",
                url="https://example.com/search",
                snippet="이 정보는 검증이 필요합니다.",
                source_domain="example.com"
            )
        ]
    
    def _extract_domain(self, url: str) -> str:
        """URL에서 도메인 추출"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.replace("www.", "")
        except:
            return ""
    
    async def _compare_with_search_results(
        self, 
        slide_data: Dict[str, Any], 
        query: str,
        search_results: List[SearchResult]
    ) -> FactVerification:
        """3단계: 검색 결과와 슬라이드 내용 대조"""
        
        if not search_results:
            return FactVerification(
                claim=query,
                is_accurate=True,  # 검색 결과 없으면 기본값
                is_outdated=False,
                confidence=0.1,
                reasoning="검색 결과가 없어 검증할 수 없습니다",
                search_results=[]
            )
        
        # 슬라이드 내용 정리
        slide_content = ""
        if slide_data.get("caption"):
            slide_content += f"캡션: {slide_data['caption']}\n"
        if slide_data.get("slide_text"):
            slide_content += f"텍스트: {slide_data['slide_text']}\n"
        
        # 검색 결과 요약
        search_summary = "\n".join([
            f"출처 {i+1} ({result.source_domain}): {result.title}\n{result.snippet}"
            for i, result in enumerate(search_results[:3])
        ])
        
        comparison_prompt = f"""다음 슬라이드 내용과 외부 검색 결과를 비교하여 사실 정확성을 판단해주세요.

슬라이드 내용:
{slide_content}

검색 결과:
{search_summary}

현재 날짜: {datetime.now().strftime('%Y년 %m월')}

다음 기준으로 판단하세요:
1. 정확성: 슬라이드의 정보가 검색 결과와 일치하는가?
2. 최신성: 슬라이드의 정보가 현재 시점에서 최신인가?

JSON 형식으로 응답해주세요:
{{
    "is_accurate": true|false,
    "is_outdated": true|false,
    "confidence": 0.0-1.0,
    "reasoning": "판단 근거 (2-3문장)"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": comparison_prompt}
                ],
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # JSON 파싱
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            
            return FactVerification(
                claim=query,
                is_accurate=result.get("is_accurate", True),
                is_outdated=result.get("is_outdated", False),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", "검증 완료"),
                search_results=search_results
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"대조 결과 JSON 파싱 실패: {response_text[:100]}... - {str(e)}")
            return FactVerification(
                claim=query,
                is_accurate=True,
                is_outdated=False,
                confidence=0.1,
                reasoning="결과 파싱 실패",
                search_results=search_results
            )
        except Exception as e:
            logger.error(f"검색 결과 대조 실패: {str(e)}")
            return FactVerification(
                claim=query,
                is_accurate=True,
                is_outdated=False,
                confidence=0.1,
                reasoning="대조 과정 오류",
                search_results=search_results
            )
    
    def _create_fact_issue(
        self, 
        slide_data: Dict[str, Any], 
        verification: FactVerification,
        doc_meta: DocumentMeta
    ) -> Issue:
        """4단계: 팩트체킹 이슈 생성"""
        
        # 이슈 메시지 구성
        if not verification.is_accurate:
            message = f"사실 정확성 의심: {verification.reasoning}"
            suggestion = "외부 출처를 확인하여 정보를 검증하세요."
        else:  # is_outdated
            message = f"정보 최신성 문제: {verification.reasoning}"
            suggestion = "최신 데이터로 업데이트를 고려하세요."
        
        # 텍스트 위치는 더미로 설정 (캡션 기반이므로 정확한 위치 파악 어려움)
        text_location = TextLocation(start=0, end=len(verification.claim))
        
        issue_id = generate_issue_id(
            doc_meta.doc_id,
            slide_data["page_id"],
            text_location,
            IssueType.FACT
        )
        
        return Issue(
            issue_id=issue_id,
            doc_id=doc_meta.doc_id,
            page_id=slide_data["page_id"],
            issue_type=IssueType.FACT,
            text_location=text_location,
            bbox_location=None,
            element_id=None,
            original_text=verification.claim,
            message=message,
            suggestion=suggestion,
            confidence=verification.confidence,
            confidence_level="medium",
            agent_name="fact_check_agent"
        )
    
    def get_factcheck_summary(self, issues: List[Issue]) -> Dict[str, Any]:
        """팩트체킹 결과 요약"""
        if not issues:
            return {
                "total_fact_issues": 0,
                "accuracy_issues": 0,
                "outdated_issues": 0,
                "avg_confidence": 0.0,
                "recommendations": ["팩트체킹에서 문제가 발견되지 않았습니다."]
            }
        
        # 이슈 분류
        accuracy_issues = 0
        outdated_issues = 0
        total_confidence = 0
        
        for issue in issues:
            if "정확성" in issue.message:
                accuracy_issues += 1
            elif "최신성" in issue.message:
                outdated_issues += 1
            
            total_confidence += issue.confidence
        
        avg_confidence = total_confidence / len(issues) if issues else 0
        
        # 권장사항
        recommendations = []
        if accuracy_issues > 0:
            recommendations.append(f"사실 정확성 검토가 필요한 항목이 {accuracy_issues}개 있습니다.")
        if outdated_issues > 0:
            recommendations.append(f"정보 업데이트가 필요한 항목이 {outdated_issues}개 있습니다.")
        
        return {
            "total_fact_issues": len(issues),
            "accuracy_issues": accuracy_issues,
            "outdated_issues": outdated_issues,
            "avg_confidence": avg_confidence,
            "recommendations": recommendations or ["팩트체킹 완료"]
        }


# E2E 테스트 함수
async def test_fact_check_agent_e2e():
    """FactCheckAgent E2E 테스트"""
    print("🧪 FactCheckAgent E2E 테스트 시작...")
    
    import os
    from dotenv import load_dotenv
    from pathlib import Path
    
    env_path = Path(__file__).resolve().parents[2] / '.env.dev'
    load_dotenv(env_path)
    
    openai_key = os.getenv("OPENAI_API_KEY")
    serpapi_key = os.getenv("SERPAPI_API_KEY")  # 선택사항
    
    if not openai_key:
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        return
    
    try:
        # 1. DocumentAgent로 문서 처리
        print("📖 DocumentAgent로 문서 처리 중...")
        from src.agents.document_agent import DocumentAgent
        
        document_agent = DocumentAgent(
            openai_api_key=openai_key,
            vision_model="gpt-5-nano"
        )
        
        # 테스트 파일 찾기
        test_files = ["sample_docs/sample.pdf"]
        test_file = None
        
        for file_name in test_files:
            if Path(file_name).exists():
                test_file = file_name
                break
        
        if not test_file:
            print("❌ 테스트할 PDF 파일이 없습니다.")
            print("   Mock 데이터로 테스트를 진행합니다.")
            
            # Mock DocumentAgent
            class MockDocumentAgent:
                def get_document(self, doc_id):
                    from src.core.models import DocumentMeta
                    return DocumentMeta(
                        doc_id=doc_id,
                        title="팩트체크 테스트 문서",
                        doc_type="pdf",
                        total_pages=3,
                        file_path="test.pdf"
                    )
                
                def get_slide_data(self, doc_id):
                    return [
                        {
                            "doc_id": doc_id,
                            "page_id": "p001",
                            "page_number": 1,
                            "caption": "GPT-4는 2023년에 OpenAI에서 출시한 대규모 언어 모델입니다. 사용자 수는 1억명을 넘었습니다.",
                            "slide_text": "GPT-4 소개\n- 출시: 2023년\n- 개발사: OpenAI",
                            "dimensions": (1920, 1080),
                            "size_bytes": 123456
                        },
                        {
                            "doc_id": doc_id,
                            "page_id": "p002",
                            "page_number": 2,
                            "caption": "머신러닝에서 경사하강법은 θ = θ - η∇J(θ) 공식으로 표현됩니다.",
                            "slide_text": "경사하강법\n- 수식: θ = θ - η∇J(θ)\n- η: 학습률",
                            "dimensions": (1920, 1080),
                            "size_bytes": 98765
                        },
                        {
                            "doc_id": doc_id,
                            "page_id": "p003",
                            "page_number": 3,
                            "caption": "2024년 한국의 AI 시장 규모는 5조원에 달할 것으로 예상됩니다. 정부 정책에 따라 변동 가능합니다.",
                            "slide_text": "AI 시장 전망\n- 2024년 예상: 5조원\n- 정부 정책 영향",
                            "dimensions": (1920, 1080),
                            "size_bytes": 87654
                        }
                    ]
            
            document_agent = MockDocumentAgent()
            doc_meta = document_agent.get_document("mock_doc_001")
            
        else:
            # 실제 파일 처리
            print(f"   파일: {test_file}")
            doc_meta = await document_agent.process_document(test_file)
        
        print(f"✅ 문서 처리 완료: {doc_meta.doc_id}")
        
        # 2. FactCheckAgent로 팩트체킹
        print("\n🔍 FactCheckAgent로 팩트체킹 중...")
        
        fact_agent = FactCheckAgent(
            openai_api_key=openai_key,
            serpapi_key=serpapi_key,
            model="gpt-5-nano"
        )
        
        issues = await fact_agent.analyze_document(document_agent, doc_meta.doc_id)
        
        print(f"✅ 팩트체킹 완료!")
        print(f"   발견된 이슈: {len(issues)}개")
        
        # 3. 결과 분석
        if issues:
            print(f"\n📋 팩트체킹 이슈들:")
            
            for i, issue in enumerate(issues, 1):
                print(f"\n{i}. [{issue.issue_type.value.upper()}] {issue.page_id}")
                print(f"   원본: {issue.original_text[:60]}...")
                print(f"   문제: {issue.message}")
                print(f"   제안: {issue.suggestion}")
                print(f"   신뢰도: {issue.confidence:.2f}")
        else:
            print("\n✅ 팩트체킹 이슈가 발견되지 않았습니다!")
        
        # 4. 요약 정보
        summary = fact_agent.get_factcheck_summary(issues)
        print(f"\n📊 팩트체킹 요약:")
        print(f"   총 이슈: {summary['total_fact_issues']}개")
        print(f"   정확성 문제: {summary['accuracy_issues']}개")
        print(f"   최신성 문제: {summary['outdated_issues']}개")
        print(f"   평균 신뢰도: {summary['avg_confidence']:.2f}")
        
        print(f"\n🎯 권장사항:")
        for rec in summary['recommendations']:
            print(f"   - {rec}")
        
        # 5. 캐시 정보
        print(f"\n💾 캐시 상태:")
        print(f"   검색 캐시: {len(fact_agent.search_cache)}개")
        print(f"   검증 캐시: {len(fact_agent.verification_cache)}개")
        
    except ImportError as e:
        print(f"❌ 모듈 임포트 실패: {str(e)}")
        print("   DocumentAgent 클래스의 임포트 경로를 확인해주세요.")
        
    except Exception as e:
        print(f"❌ E2E 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 FactCheckAgent E2E 테스트 완료!")


# 통합 테스트 함수
async def test_full_pipeline():
    """전체 파이프라인 통합 테스트 (DocumentAgent + QualityAgent + FactCheckAgent)"""
    print("🧪 전체 파이프라인 통합 테스트 시작...")
    
    import os
    from dotenv import load_dotenv
    from pathlib import Path
    
    env_path = Path(__file__).resolve().parents[2] / '.env.dev'
    load_dotenv(env_path)
    
    openai_key = os.getenv("OPENAI_API_KEY")
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    
    if not openai_key:
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        return
    
    try:
        # 테스트용 Mock DocumentAgent (간단한 데이터)
        class MockDocumentAgent:
            def get_document(self, doc_id):
                from src.core.models import DocumentMeta
                return DocumentMeta(
                    doc_id=doc_id,
                    title="EDU-Audit 통합 테스트",
                    doc_type="pdf",
                    total_pages=2,
                    file_path="integration_test.pdf"
                )
            
            def get_slide_data(self, doc_id):
                return [
                    {
                        "doc_id": doc_id,
                        "page_id": "p001",
                        "page_number": 1,
                        "caption": "ChatGPT는 2022년에 출시되었고 현재 사용자 수가 1억명을 넘었습니다. 이는 최신 통계입니다.",
                        "slide_text": "ChatGPT 현황\n- 출시: 2022년\n- 사용자: 1억명+",
                        "image_base64": "dummy_base64_data",
                        "dimensions": (1920, 1080),
                        "size_bytes": 123456
                    },
                    {
                        "doc_id": doc_id,
                        "page_id": "p002", 
                        "page_number": 2,
                        "caption": "머신러닝에서 경사하강법은 θ = θ - α∇J(θ) 공식으로 표현됩니다. 여기서 α는 학습률입니다.",
                        "slide_text": "경사하강법 공식\nθ = θ - α∇J(θ)",
                        "image_base64": "dummy_base64_data",
                        "dimensions": (1920, 1080),
                        "size_bytes": 98765
                    }
                ]
        
        document_agent = MockDocumentAgent()
        doc_id = "integration_test_001"
        
        print("📖 Mock 문서 데이터 준비 완료")
        
        # 1. QualityAgent 실행
        print("\n🔍 QualityAgent 실행 중...")
        try:
            from src.agents.quality_agent import QualityAgent, QualityConfig
            
            quality_config = QualityConfig(
                max_issues_per_slide=2,
                confidence_threshold=0.7,
                issue_severity_filter="medium"
            )
            
            quality_agent = QualityAgent(
                openai_api_key=openai_key,
                vision_model="gpt-5-nano",
                config=quality_config
            )
            
            quality_issues = await quality_agent.analyze_document(document_agent, doc_id)
            print(f"   품질 이슈: {len(quality_issues)}개")
            
        except ImportError:
            print("   ⚠️ QualityAgent 임포트 실패 - 건너뜀")
            quality_issues = []
        except Exception as e:
            print(f"   ❌ QualityAgent 실행 실패: {str(e)}")
            quality_issues = []
        
        # 2. FactCheckAgent 실행  
        print("\n🔍 FactCheckAgent 실행 중...")
        
        fact_agent = FactCheckAgent(
            openai_api_key=openai_key,
            serpapi_key=serpapi_key,
            model="gpt-5-nano"
        )
        
        fact_issues = await fact_agent.analyze_document(document_agent, doc_id)
        print(f"   팩트체킹 이슈: {len(fact_issues)}개")
        
        # 3. 통합 결과 분석
        all_issues = quality_issues + fact_issues
        
        print(f"\n📊 통합 분석 결과:")
        print(f"   총 이슈: {len(all_issues)}개")
        print(f"   품질 이슈: {len(quality_issues)}개")
        print(f"   팩트체킹 이슈: {len(fact_issues)}개")
        
        # 이슈 타입별 분류
        issue_by_type = {}
        for issue in all_issues:
            issue_type = issue.issue_type.value
            if issue_type not in issue_by_type:
                issue_by_type[issue_type] = []
            issue_by_type[issue_type].append(issue)
        
        print(f"\n📈 이슈 타입별 분포:")
        for issue_type, issues in issue_by_type.items():
            print(f"   {issue_type}: {len(issues)}개")
            
            # 각 타입에서 대표 이슈 1개씩 출력
            if issues:
                sample_issue = issues[0]
                print(f"     예시: {sample_issue.message[:50]}...")
        
        # 4. 에이전트별 요약
        if quality_issues:
            quality_summary = quality_agent.get_quality_summary(quality_issues)
            print(f"\n🎯 품질 요약:")
            print(f"   품질 점수: {quality_summary['quality_score']:.2f}/1.0")
            for rec in quality_summary['recommendations'][:2]:
                print(f"   - {rec}")
        
        if fact_issues:
            fact_summary = fact_agent.get_factcheck_summary(fact_issues)
            print(f"\n🎯 팩트체킹 요약:")
            print(f"   평균 신뢰도: {fact_summary['avg_confidence']:.2f}")
            for rec in fact_summary['recommendations'][:2]:
                print(f"   - {rec}")
        
        # 5. 최종 권장사항
        print(f"\n✅ 최종 권장사항:")
        if len(all_issues) == 0:
            print("   문서 품질과 팩트체킹 모두 양호합니다.")
        elif len(quality_issues) > len(fact_issues):
            print("   품질 개선에 우선 집중하시기 바랍니다.")
        else:
            print("   사실 확인 및 정보 업데이트가 우선 필요합니다.")
        
    except Exception as e:
        print(f"❌ 통합 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 전체 파이프라인 통합 테스트 완료!")


# 단위 테스트 함수
async def test_fact_check_agent():
    """FactCheckAgent 단위 테스트"""
    print("🧪 FactCheckAgent 단위 테스트 시작...")
    
    import os
    from dotenv import load_dotenv
    from pathlib import Path
    
    env_path = Path(__file__).resolve().parents[2] / '.env.dev'
    load_dotenv(env_path)
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY가 필요합니다.")
        return
    
    # 에이전트 생성
    agent = FactCheckAgent(
        openai_api_key=openai_key,
        serpapi_key=None,  # 단위 테스트에서는 검색 비활성화
        model="gpt-5-nano"
    )
    
    # 테스트 슬라이드 데이터
    test_slides = [
        {
            "page_id": "p001",
            "page_number": 1,
            "caption": "GPT-4는 2023년 3월에 OpenAI가 출시한 대규모 언어 모델입니다.",
            "slide_text": "GPT-4 출시일: 2023년 3월"
        },
        {
            "page_id": "p002", 
            "page_number": 2,
            "caption": "딥러닝에서 역전파는 ∂L/∂w = δx 공식으로 계산됩니다.",
            "slide_text": "역전파 공식: ∂L/∂w = δx"
        },
        {
            "page_id": "p003",
            "page_number": 3, 
            "caption": "2024년 한국 AI 투자 규모는 10조원을 돌파했습니다.",
            "slide_text": "AI 투자: 10조원 돌파"
        }
    ]
    
    print(f"📋 {len(test_slides)}개 슬라이드 트리거 테스트...")
    
    # 1. 트리거 테스트
    async with agent:
        for slide in test_slides:
            trigger = await agent._check_factcheck_trigger(slide)
            
            print(f"\n슬라이드 {slide['page_id']}:")
            print(f"   팩트체크 필요: {'✅' if trigger.factcheck_required else '❌'}")
            print(f"   이유: {trigger.reason}")
            print(f"   키워드: {trigger.keywords}")
            print(f"   신뢰도: {trigger.confidence:.2f}")
    
    print("\n🎉 단위 테스트 완료!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "e2e":
            asyncio.run(test_fact_check_agent_e2e())
        elif sys.argv[1] == "pipeline":
            asyncio.run(test_full_pipeline()) 
        elif sys.argv[1] == "unit":
            asyncio.run(test_fact_check_agent())
        else:
            print("사용법: python fact_check_agent.py [e2e|pipeline|unit]")
    else:
        asyncio.run(test_fact_check_agent())