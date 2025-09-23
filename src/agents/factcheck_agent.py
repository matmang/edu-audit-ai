"""
EDU-Audit Fact Check Agent
사실 검증 및 정보 최신성 검사 에이전트
"""

import asyncio
import logging
import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from urllib.parse import quote

import aiohttp
from llama_index.llms.openai import OpenAI

from src.core.models import (
    DocumentMeta, PageInfo, PageElement, ElementType, Issue, IssueType,
    FactCheckRequest, FactCheckResult, TextLocation, BoundingBox,
    generate_issue_id
)

logger = logging.getLogger(__name__)


@dataclass
class FactCheckRule:
    """사실 검증 규칙"""
    domains: List[str]  # 적용 도메인 (science, technology, history, etc.)
    claim_patterns: List[str]  # 검증이 필요한 문장 패턴
    confidence_threshold: float = 0.7  # 신뢰도 임계값
    require_sources: bool = True  # 출처 필요 여부


@dataclass
class SearchResult:
    """검색 결과"""
    title: str
    url: str
    snippet: str
    published_date: Optional[str] = None
    source_domain: str = ""
    relevance_score: float = 0.0


@dataclass
class FactVerification:
    """사실 검증 결과"""
    claim: str
    is_factual: bool
    confidence: float
    evidence: List[SearchResult]
    reasoning: str
    is_outdated: bool = False
    last_updated: Optional[str] = None
    contradictory_info: Optional[str] = None


class FactCheckAgent:
    """사실 검증 에이전트"""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        serpapi_key: Optional[str] = None,
        llm_model: str = "gpt-5-nano",
        max_search_results: int = 5,
        fact_check_timeout: int = 30
    ):
        self.openai_api_key = openai_api_key
        self.serpapi_key = serpapi_key
        self.max_search_results = max_search_results
        self.fact_check_timeout = fact_check_timeout
        
        # LLM 초기화
        self.llm = None
        if openai_api_key:
            self.llm = OpenAI(
                model=llm_model,
                temperature=0.1,  # 일관된 분석을 위해 낮게 설정
                api_key=openai_api_key
            )
        
        # HTTP 클라이언트
        self.session = None
        
        # 사실 검증 규칙 로드
        self._load_fact_check_rules()
        
        # 신뢰할 수 있는 소스 목록
        self._load_trusted_sources()
        
        logger.info("FactCheckAgent 초기화 완료")
        if not openai_api_key:
            logger.warning("OpenAI API 키가 없습니다. LLM 분석이 제한됩니다.")
        if not serpapi_key:
            logger.warning("SerpAPI 키가 없습니다. 웹 검색이 제한됩니다.")
    
    def _load_fact_check_rules(self):
        """사실 검증 규칙 로드"""
        self.fact_check_rules = [
            FactCheckRule(
                domains=["science", "technology", "research"],
                claim_patterns=[
                    r"연구에 따르면",
                    r"최신 연구",
                    r"[0-9]{4}년.*연구",
                    r"실험 결과",
                    r"과학자들은.*발견",
                    r"통계에 의하면",
                    r"데이터에 따르면"
                ],
                confidence_threshold=0.8,
                require_sources=True
            ),
            FactCheckRule(
                domains=["statistics", "data", "numbers"],
                claim_patterns=[
                    r"[0-9]+%",
                    r"[0-9,]+명",
                    r"[0-9,]+개",
                    r"순위.*[0-9]+위",
                    r"증가.*[0-9]+%",
                    r"감소.*[0-9]+%"
                ],
                confidence_threshold=0.7,
                require_sources=True
            ),
            FactCheckRule(
                domains=["technology", "software", "ai"],
                claim_patterns=[
                    r"최신 버전",
                    r"새로운 기능",
                    r"업데이트",
                    r"[0-9]{4}년.*출시",
                    r"현재.*지원",
                    r"GPT-[0-9]+",
                    r"ChatGPT",
                    r"최신.*모델"
                ],
                confidence_threshold=0.6,
                require_sources=False
            ),
            FactCheckRule(
                domains=["events", "news", "current"],
                claim_patterns=[
                    r"최근.*발표",
                    r"올해",
                    r"이번 달",
                    r"현재.*상황",
                    r"작년",
                    r"[0-9]{4}년.*월"
                ],
                confidence_threshold=0.8,
                require_sources=True
            )
        ]
    
    def _load_trusted_sources(self):
        """신뢰할 수 있는 소스 목록 로드"""
        self.trusted_sources = {
            # 학술/연구 소스
            "academic": [
                "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "scholar.google.com",
                "ieee.org", "acm.org", "nature.com", "science.org",
                "springer.com", "elsevier.com"
            ],
            # 뉴스/미디어 소스
            "news": [
                "bbc.com", "reuters.com", "ap.org", "nytimes.com",
                "washingtonpost.com", "cnn.com", "npr.org"
            ],
            # 기술 소스
            "technology": [
                "github.com", "stackoverflow.com", "developer.mozilla.org",
                "techcrunch.com", "arstechnica.com", "wired.com"
            ],
            # 정부/공식 소스
            "official": [
                ".gov", ".edu", "who.int", "unesco.org", "un.org"
            ]
        }
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.fact_check_timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    async def check_document_facts(self, doc_meta: DocumentMeta, pages: List[PageInfo]) -> List[Issue]:
        """
        문서 전체 사실 검증
        
        Args:
            doc_meta: 문서 메타데이터
            pages: 페이지 목록
            
        Returns:
            List[Issue]: 발견된 사실 오류 이슈들
        """
        logger.info(f"문서 사실 검증 시작: {doc_meta.doc_id}")
        
        all_issues = []
        
        async with self:
            for page in pages:
                # 1. 페이지 텍스트에서 검증 대상 추출
                text_claims = await self._extract_fact_claims(page.raw_text)
                
                for claim in text_claims:
                    page_issues = await self._verify_claim_in_page(
                        doc_meta, page, claim, page.raw_text
                    )
                    all_issues.extend(page_issues)
                
                # 2. 멀티모달 요소에서 검증 대상 추출
                for element in page.elements:
                    element_issues = await self._verify_multimodal_element(
                        doc_meta, page, element
                    )
                    all_issues.extend(element_issues)
                
                # API 호출 제한
                await asyncio.sleep(0.5)
        
        logger.info(f"사실 검증 완료: {len(all_issues)}개 이슈 발견")
        return all_issues
    
    async def verify_single_claim(self, claim: str, context: str = None) -> FactCheckResult:
        """
        단일 사실 주장 검증
        
        Args:
            claim: 검증할 주장
            context: 문맥 정보
            
        Returns:
            FactCheckResult: 검증 결과
        """
        logger.info(f"단일 사실 검증: {claim[:50]}...")
        
        async with self:
            # 1. 웹 검색으로 관련 정보 수집
            search_results = await self._search_for_claim(claim)
            
            # 2. LLM을 통한 사실 검증
            verification = await self._verify_with_llm(claim, search_results, context)
            
            # 3. 최신성 검사
            is_outdated, last_updated = await self._check_if_outdated(claim, search_results)
            
            result = FactCheckResult(
                sentence=claim,
                is_factual=verification.is_factual,
                confidence=verification.confidence,
                explanation=verification.reasoning,
                sources=[result.url for result in verification.evidence],
                checked_at=datetime.now()
            )
            
            # 추가 정보
            if is_outdated:
                result.explanation += f"\n\n⚠️ 주의: 이 정보는 오래된 것일 수 있습니다 (마지막 업데이트: {last_updated})"
            
            if verification.contradictory_info:
                result.explanation += f"\n\n🔄 상충 정보: {verification.contradictory_info}"
            
            return result
    
    async def _extract_fact_claims(self, text: str) -> List[str]:
        """텍스트에서 사실 검증이 필요한 주장들 추출"""
        claims = []
        
        if not text.strip():
            return claims
        
        # 문장 단위로 분할
        sentences = re.split(r'[.!?]\s+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # 너무 짧은 문장은 제외
                continue
            
            # 각 규칙에 대해 패턴 매칭
            for rule in self.fact_check_rules:
                for pattern in rule.claim_patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        claims.append(sentence)
                        break
        
        # 중복 제거
        return list(set(claims))
    
    async def _verify_claim_in_page(
        self, 
        doc_meta: DocumentMeta, 
        page: PageInfo, 
        claim: str,
        full_text: str
    ) -> List[Issue]:
        """페이지 내 특정 주장 검증"""
        issues = []
        
        try:
            # 주장 검증 수행
            verification = await self._perform_fact_verification(claim)
            
            if not verification.is_factual or verification.is_outdated:
                # 원본 텍스트에서 위치 찾기
                pos = full_text.find(claim)
                if pos != -1:
                    location = TextLocation(start=pos, end=pos + len(claim))
                else:
                    # 부분 매칭 시도
                    words = claim.split()[:5]  # 처음 5단어로 검색
                    partial_claim = " ".join(words)
                    pos = full_text.find(partial_claim)
                    if pos != -1:
                        location = TextLocation(start=pos, end=pos + len(partial_claim))
                    else:
                        location = TextLocation(start=0, end=len(claim))
                
                # 이슈 타입 결정
                if not verification.is_factual:
                    issue_message = f"사실 오류 가능성: {verification.reasoning}"
                    suggestion = "출처 확인 및 최신 정보 업데이트 필요"
                else:  # is_outdated
                    issue_message = f"정보가 오래되었을 수 있음: {verification.last_updated}"
                    suggestion = "최신 정보로 업데이트 권장"
                
                issue_id = generate_issue_id(
                    doc_meta.doc_id,
                    page.page_id,
                    location,
                    IssueType.FACT
                )
                
                issue = Issue(
                    issue_id=issue_id,
                    doc_id=doc_meta.doc_id,
                    page_id=page.page_id,
                    issue_type=IssueType.FACT,
                    text_location=location,
                    original_text=claim,
                    message=issue_message,
                    suggestion=suggestion,
                    confidence=verification.confidence,
                    confidence_level="high",  # Pydantic이 자동 계산
                    agent_name="fact_check_agent"
                )
                
                issues.append(issue)
        
        except Exception as e:
            logger.warning(f"주장 검증 실패 '{claim[:30]}...': {str(e)}")
        
        return issues
    
    async def _verify_multimodal_element(
        self, 
        doc_meta: DocumentMeta, 
        page: PageInfo, 
        element: PageElement
    ) -> List[Issue]:
        """멀티모달 요소의 사실 검증"""
        issues = []
        
        try:
            text_to_verify = None
            
            # 요소 타입별 텍스트 추출
            if element.element_type == ElementType.IMAGE and element.image_data:
                # OCR 텍스트와 AI 설명에서 검증 대상 찾기
                if element.image_data.ocr_text:
                    claims = await self._extract_fact_claims(element.image_data.ocr_text)
                    for claim in claims:
                        verification = await self._perform_fact_verification(claim)
                        if not verification.is_factual or verification.is_outdated:
                            issue = self._create_multimodal_fact_issue(
                                doc_meta, page, element, claim, verification
                            )
                            issues.append(issue)
                
                if element.image_data.description:
                    claims = await self._extract_fact_claims(element.image_data.description)
                    for claim in claims:
                        verification = await self._perform_fact_verification(claim)
                        if not verification.is_factual or verification.is_outdated:
                            issue = self._create_multimodal_fact_issue(
                                doc_meta, page, element, claim, verification
                            )
                            issues.append(issue)
            
            elif element.element_type == ElementType.TABLE and element.table_data:
                # 표 내용에서 숫자 데이터 검증
                table_text = self._extract_table_text(element.table_data)
                claims = await self._extract_fact_claims(table_text)
                for claim in claims:
                    verification = await self._perform_fact_verification(claim)
                    if not verification.is_factual or verification.is_outdated:
                        issue = self._create_multimodal_fact_issue(
                            doc_meta, page, element, claim, verification
                        )
                        issues.append(issue)
            
            elif element.element_type == ElementType.CHART and element.chart_data:
                # 차트 설명에서 데이터 관련 주장 검증
                if element.chart_data.description:
                    claims = await self._extract_fact_claims(element.chart_data.description)
                    for claim in claims:
                        verification = await self._perform_fact_verification(claim)
                        if not verification.is_factual or verification.is_outdated:
                            issue = self._create_multimodal_fact_issue(
                                doc_meta, page, element, claim, verification
                            )
                            issues.append(issue)
        
        except Exception as e:
            logger.warning(f"멀티모달 요소 검증 실패 {element.element_id}: {str(e)}")
        
        return issues
    
    def _extract_table_text(self, table_data) -> str:
        """표 데이터를 텍스트로 변환"""
        texts = []
        
        if table_data.headers:
            texts.append(" ".join(table_data.headers))
        
        for row in table_data.rows:
            texts.append(" ".join(row))
        
        return " ".join(texts)
    
    def _create_multimodal_fact_issue(
        self, 
        doc_meta: DocumentMeta, 
        page: PageInfo, 
        element: PageElement,
        claim: str, 
        verification: FactVerification
    ) -> Issue:
        """멀티모달 요소의 사실 오류 이슈 생성"""
        
        if not verification.is_factual:
            message = f"멀티모달 요소 내 사실 오류: {verification.reasoning}"
            suggestion = "출처 확인 및 정보 검증 필요"
        else:  # is_outdated
            message = f"멀티모달 요소 내 정보 과시: {verification.last_updated}"
            suggestion = "최신 데이터로 업데이트 권장"
        
        issue_id = generate_issue_id(
            doc_meta.doc_id,
            page.page_id,
            element.bbox or BoundingBox(x=0, y=0, width=100, height=100),
            IssueType.FACT
        )
        
        return Issue(
            issue_id=issue_id,
            doc_id=doc_meta.doc_id,
            page_id=page.page_id,
            issue_type=IssueType.FACT,
            bbox_location=element.bbox,
            element_id=element.element_id,
            original_text=claim,
            message=message,
            suggestion=suggestion,
            confidence=verification.confidence,
            confidence_level="high",
            agent_name="fact_check_agent"
        )
    
    async def _perform_fact_verification(self, claim: str) -> FactVerification:
        """실제 사실 검증 수행"""
        try:
            # 1. 웹 검색
            search_results = await self._search_for_claim(claim)
            
            # 2. LLM 검증
            verification = await self._verify_with_llm(claim, search_results)
            
            # 3. 최신성 검사
            is_outdated, last_updated = await self._check_if_outdated(claim, search_results)
            verification.is_outdated = is_outdated
            verification.last_updated = last_updated
            
            return verification
            
        except Exception as e:
            logger.warning(f"사실 검증 실패: {str(e)}")
            return FactVerification(
                claim=claim,
                is_factual=True,  # 검증 실패 시 기본값
                confidence=0.1,
                evidence=[],
                reasoning="검증 과정에서 오류 발생",
                is_outdated=False
            )
    
    async def _search_for_claim(self, claim: str) -> List[SearchResult]:
        """주장에 대한 웹 검색 수행"""
        search_results = []
        
        try:
            if self.serpapi_key:
                # SerpAPI를 통한 Google 검색
                results = await self._search_with_serpapi(claim)
                search_results.extend(results)
            else:
                # 대체 검색 방법 또는 더미 결과
                logger.warning("SerpAPI 키가 없어 검색을 수행할 수 없습니다")
                search_results = self._generate_dummy_search_results(claim)
        
        except Exception as e:
            logger.warning(f"웹 검색 실패: {str(e)}")
            search_results = self._generate_dummy_search_results(claim)
        
        return search_results[:self.max_search_results]
    
    async def _search_with_serpapi(self, query: str) -> List[SearchResult]:
        """SerpAPI를 통한 Google 검색"""
        if not self.session:
            return []
        
        try:
            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "api_key": self.serpapi_key,
                "engine": "google",
                "num": self.max_search_results
            }
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                results = []
                for item in data.get("organic_results", []):
                    result = SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source_domain=self._extract_domain(item.get("link", "")),
                        relevance_score=0.8  # 기본값
                    )
                    results.append(result)
                
                return results
        
        except Exception as e:
            logger.warning(f"SerpAPI 검색 실패: {str(e)}")
            return []
    
    def _generate_dummy_search_results(self, claim: str) -> List[SearchResult]:
        """더미 검색 결과 생성 (테스트용)"""
        return [
            SearchResult(
                title=f"Fact-checking: {claim[:30]}...",
                url="https://example.com/fact-check",
                snippet="이 주장에 대한 검증이 필요합니다.",
                source_domain="example.com",
                relevance_score=0.5
            )
        ]
    
    def _extract_domain(self, url: str) -> str:
        """URL에서 도메인 추출"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return ""
    
    async def _verify_with_llm(
        self, 
        claim: str, 
        search_results: List[SearchResult], 
        context: str = None
    ) -> FactVerification:
        """LLM을 통한 사실 검증"""
        if not self.llm:
            return FactVerification(
                claim=claim,
                is_factual=True,
                confidence=0.1,
                evidence=search_results,
                reasoning="LLM을 사용할 수 없어 검증하지 못했습니다"
            )
        
        try:
            # 검색 결과를 문맥으로 정리
            evidence_text = "\n".join([
                f"출처 {i+1}: {result.title}\n{result.snippet}\n"
                for i, result in enumerate(search_results)
            ])
            
            prompt = self._create_fact_verification_prompt(claim, evidence_text, context)
            
            response = await self.llm.acomplete(prompt)
            
            # LLM 응답 파싱
            verification = self._parse_llm_verification_response(
                response.text, claim, search_results
            )
            
            return verification
            
        except Exception as e:
            logger.warning(f"LLM 사실 검증 실패: {str(e)}")
            return FactVerification(
                claim=claim,
                is_factual=True,
                confidence=0.1,
                evidence=search_results,
                reasoning="LLM 검증 중 오류 발생"
            )
    
    def _create_fact_verification_prompt(self, claim: str, evidence: str, context: str = None) -> str:
        """사실 검증용 프롬프트 생성"""
        context_part = f"\n\n문맥 정보:\n{context}" if context else ""
        
        return f"""다음 주장의 사실 여부를 검증해주세요.

검증할 주장: {claim}

참고 자료:
{evidence}{context_part}

다음 형식으로 응답해주세요:
판정: [사실/거짓/불분명]
신뢰도: [0.0-1.0]
근거: [판정 이유를 자세히 설명]
상충정보: [있다면 언급, 없으면 "없음"]

주의사항:
- 검색 결과와 주장을 신중히 비교 분석하세요
- 출처의 신뢰성도 고려하세요
- 불확실한 경우 "불분명"으로 판정하세요
- 교육 자료의 정확성이 중요하므로 엄격하게 검증하세요"""
    
    def _parse_llm_verification_response(
        self, 
        response: str, 
        claim: str, 
        search_results: List[SearchResult]
    ) -> FactVerification:
        """LLM 검증 응답 파싱"""
        try:
            lines = response.strip().split('\n')
            
            # 기본값
            is_factual = True
            confidence = 0.5
            reasoning = "응답 파싱 실패"
            contradictory_info = None
            
            for line in lines:
                line = line.strip()
                
                if line.startswith("판정:"):
                    judgment = line.split(":", 1)[1].strip().lower()
                    if "거짓" in judgment or "false" in judgment:
                        is_factual = False
                    elif "불분명" in judgment or "unclear" in judgment:
                        is_factual = False
                        confidence = 0.3
                
                elif line.startswith("신뢰도:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except:
                        pass
                
                elif line.startswith("근거:"):
                    reasoning = line.split(":", 1)[1].strip()
                
                elif line.startswith("상충정보:"):
                    contradictory_part = line.split(":", 1)[1].strip()
                    if contradictory_part and contradictory_part != "없음":
                        contradictory_info = contradictory_part
            
            return FactVerification(
                claim=claim,
                is_factual=is_factual,
                confidence=confidence,
                evidence=search_results,
                reasoning=reasoning,
                contradictory_info=contradictory_info
            )
            
        except Exception as e:
            logger.warning(f"LLM 응답 파싱 실패: {str(e)}")
            return FactVerification(
                claim=claim,
                is_factual=True,
                confidence=0.1,
                evidence=search_results,
                reasoning="응답 파싱 중 오류 발생"
            )
    
    async def _check_if_outdated(
        self, 
        claim: str, 
        search_results: List[SearchResult]
    ) -> Tuple[bool, Optional[str]]:
        """정보의 최신성 검사"""
        if not self.llm:
            return False, None
        
        try:
            # 시간 관련 키워드 검사
            time_keywords = ["최신", "현재", "올해", "작년", "최근", "새로운", "업데이트"]
            has_time_reference = any(keyword in claim for keyword in time_keywords)
            
            if not has_time_reference:
                return False, None
            
            # 검색 결과에서 날짜 정보 수집
            recent_info = []
            for result in search_results:
                if result.published_date:
                    recent_info.append(f"{result.title}: {result.published_date}")
            
            if not recent_info:
                return False, None
            
            # LLM으로 최신성 분석
            prompt = f"""다음 주장이 현재 시점에서 최신 정보인지 분석해주세요.

주장: {claim}

최근 정보:
{chr(10).join(recent_info)}

현재 날짜: {datetime.now().strftime('%Y년 %m월')}

다음 형식으로 응답해주세요:
최신성: [최신/구식/불분명]
마지막업데이트: [예상 날짜 또는 "불명"]
이유: [판단 근거]"""
            
            response = await self.llm.acomplete(prompt)
            
            # 응답 파싱
            lines = response.text.strip().split('\n')
            is_outdated = False
            last_updated = None
            
            for line in lines:
                if line.startswith("최신성:"):
                    status = line.split(":", 1)[1].strip().lower()
                    if "구식" in status or "outdated" in status:
                        is_outdated = True
                
                elif line.startswith("마지막업데이트:"):
                    last_updated = line.split(":", 1)[1].strip()
                    if last_updated == "불명":
                        last_updated = None
            
            return is_outdated, last_updated
            
        except Exception as e:
            logger.warning(f"최신성 검사 실패: {str(e)}")
            return False, None
    
    def get_source_credibility_score(self, url: str) -> float:
        """소스의 신뢰도 점수 계산"""
        domain = self._extract_domain(url)
        
        # 신뢰할 수 있는 소스 체크
        for category, sources in self.trusted_sources.items():
            for trusted_source in sources:
                if trusted_source in domain:
                    if category == "academic":
                        return 0.9
                    elif category == "official":
                        return 0.85
                    elif category == "news":
                        return 0.8
                    elif category == "technology":
                        return 0.75
        
        # 일반 웹사이트
        if domain.endswith('.edu') or domain.endswith('.gov'):
            return 0.85
        elif domain.endswith('.org'):
            return 0.7
        else:
            return 0.5


# 테스트 함수
async def test_fact_check_agent():
    """FactCheckAgent 테스트"""
    print("🧪 FactCheckAgent 테스트 시작...")
    
    import os
    from dotenv import load_dotenv
    from pathlib import Path
    env_path = Path(__file__).resolve().parents[2] / '.env.dev'
    load_dotenv(env_path)
    
    openai_key = os.getenv("OPENAI_API_KEY")
    serpapi_key = os.getenv("SERPAPI_API_KEY")  # 필요 시 추가
    
    # 에이전트 생성
    agent = FactCheckAgent(
        openai_api_key=openai_key,
        serpapi_key=serpapi_key
    )
    
    # 테스트 주장들
    test_claims = [
        "GPT-4는 2023년에 출시되었습니다",
        "한국의 인구는 약 5천만명입니다",
        "최신 연구에 따르면 딥러닝 모델의 정확도가 95%를 넘었습니다",
        "ChatGPT는 현재 무료로 사용할 수 있습니다",
        "작년 AI 시장 규모는 100조원을 넘었습니다"
    ]
    
    print(f"\n🔍 {len(test_claims)}개 주장 검증 중...")
    
    for i, claim in enumerate(test_claims, 1):
        print(f"\n[{i}] 검증 중: {claim}")
        
        try:
            result = await agent.verify_single_claim(claim)
            
            print(f"    결과: {'✅ 사실' if result.is_factual else '❌ 거짓/의심'}")
            print(f"    신뢰도: {result.confidence:.2f}")
            print(f"    근거: {result.explanation[:100]}...")
            if result.sources:
                print(f"    출처: {len(result.sources)}개")
        
        except Exception as e:
            print(f"    ❌ 검증 실패: {str(e)}")
    
    # 멀티모달 문서 테스트 (더미)
    print(f"\n📄 멀티모달 문서 사실 검증 테스트...")
    
    from src.core.models import DocumentMeta, PageInfo, PageElement, ElementType, ImageElement, generate_doc_id
    
    # 테스트용 문서 생성
    doc_meta = DocumentMeta(
        doc_id=generate_doc_id("fact_test.pdf"),
        title="사실 검증 테스트 문서",
        doc_type="pdf",
        total_pages=1,
        file_path="fact_test.pdf"
    )
    
    # 사실 검증 대상이 포함된 페이지
    test_page = PageInfo(
        page_id="p001",
        page_number=1,
        raw_text="최신 연구에 따르면 GPT-4의 성능이 이전 모델보다 40% 향상되었습니다. 2024년 현재 AI 시장은 급성장하고 있습니다.",
        word_count=25,
        elements=[]
    )
    
    print("    문서 내 사실 검증 실행 중...")
    
    try:
        issues = await agent.check_document_facts(doc_meta, [test_page])
        
        print(f"    발견된 사실 이슈: {len(issues)}개")
        
        for issue in issues:
            print(f"      - {issue.message[:80]}...")
            print(f"        신뢰도: {issue.confidence:.2f}")
            print(f"        제안: {issue.suggestion[:60]}...")
    
    except Exception as e:
        print(f"    ❌ 문서 검증 실패: {str(e)}")
    
    print("\n🎉 FactCheckAgent 테스트 완료!")


async def test_fact_check_integration():
    """FactCheckAgent와 다른 에이전트 통합 테스트"""
    print("🧪 FactCheckAgent 통합 테스트 시작...")
    
    import os
    from dotenv import load_dotenv
    from pathlib import Path
    env_path = Path(__file__).resolve().parents[2] / '.env.dev'
    load_dotenv(env_path)
    
    # 에이전트들 초기화
    from src.agents.document_agent import MultimodalDocumentAgent
    from src.agents.quality_agent import MultimodalQualityAgent
    
    openai_key = os.getenv("OPENAI_API_KEY")
    
    document_agent = MultimodalDocumentAgent(openai_api_key=openai_key)
    quality_agent = MultimodalQualityAgent(openai_api_key=openai_key)
    fact_agent = FactCheckAgent(openai_api_key=openai_key)
    
    # 테스트용 더미 문서 생성 (실제 파일이 없는 경우)
    from src.core.models import (
        DocumentMeta, PageInfo, PageElement, ElementType, 
        ImageElement, BoundingBox, generate_doc_id
    )
    
    doc_meta = DocumentMeta(
        doc_id=generate_doc_id("integrated_test.pdf"),
        title="통합 테스트 문서",
        doc_type="pdf", 
        total_pages=1,
        file_path="integrated_test.pdf"
    )
    
    # 사실 검증이 필요한 내용을 포함한 페이지
    image_element = PageElement(
        element_id="p001_image_001",
        element_type=ElementType.IMAGE,
        bbox=BoundingBox(x=100, y=200, width=300, height=200),
        image_data=ImageElement(
            element_id="p001_image_001",
            bbox=BoundingBox(x=100, y=200, width=300, height=200),
            format="png",
            size_bytes=12345,
            dimensions=(300, 200),
            ocr_text="2023년 연구에 따르면 ChatGPT 사용자는 1억명을 넘었습니다",
            description="AI 사용자 통계 차트"
        )
    )
    
    test_page = PageInfo(
        page_id="p001",
        page_number=1,
        raw_text="최신 연구에 따르면 딥러닝 모델의 정확도가 99%에 달합니다. GPT-4는 현재 가장 강력한 언어 모델입니다.",
        word_count=20,
        elements=[image_element]
    )
    
    print("\n🔍 통합 분석 실행...")
    
    try:
        # 1. 품질 검사
        print("1. 품질 검사 실행...")
        quality_issues = await quality_agent.check_document(doc_meta, [test_page])
        print(f"   품질 이슈: {len(quality_issues)}개")
        
        # 2. 사실 검증
        print("2. 사실 검증 실행...")
        fact_issues = await fact_agent.check_document_facts(doc_meta, [test_page])
        print(f"   사실 이슈: {len(fact_issues)}개")
        
        # 3. 통합 결과 분석
        all_issues = quality_issues + fact_issues
        
        print(f"\n📊 통합 결과:")
        print(f"   총 이슈: {len(all_issues)}개")
        
        issue_by_type = {}
        for issue in all_issues:
            issue_type = issue.issue_type.value
            if issue_type not in issue_by_type:
                issue_by_type[issue_type] = []
            issue_by_type[issue_type].append(issue)
        
        for issue_type, issues in issue_by_type.items():
            print(f"   {issue_type}: {len(issues)}개")
            for issue in issues[:2]:  # 처음 2개만 출력
                print(f"     - {issue.message[:60]}...")
        
        # 4. 멀티모달 + 사실검증 특화 분석
        multimodal_fact_issues = [
            issue for issue in fact_issues 
            if issue.element_id is not None
        ]
        
        print(f"\n🖼️ 멀티모달 사실 검증:")
        print(f"   멀티모달 요소의 사실 이슈: {len(multimodal_fact_issues)}개")
        
        for issue in multimodal_fact_issues:
            print(f"     요소 {issue.element_id}: {issue.message[:50]}...")
    
    except Exception as e:
        print(f"❌ 통합 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 통합 테스트 완료!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "integration":
        asyncio.run(test_fact_check_integration())
    else:
        asyncio.run(test_fact_check_agent())