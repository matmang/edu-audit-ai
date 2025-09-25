"""
EDU-Audit Quality Agent - Simplified
슬라이드 캡션 기반 교육 품질 검수 에이전트
"""

import asyncio
import logging
import json
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from openai import AsyncOpenAI

from src.core.models import (
    DocumentMeta, Issue, IssueType, TextLocation,
    generate_issue_id
)

logger = logging.getLogger(__name__)


@dataclass
class QualityConfig:
    """품질 검수 설정"""
    max_issues_per_slide: int = 3
    confidence_threshold: float = 0.7
    enable_vision_analysis: bool = False
    issue_severity_filter: str = "medium"  # low, medium, high
    
    # 필터링할 이슈 타입들 (너무 사소한 것들)
    exclude_minor_issues: List[str] = None
    
    def __post_init__(self):
        if self.exclude_minor_issues is None:
            self.exclude_minor_issues = [
                "missing_period",  # 문장 끝 마침표 없음
                "whitespace_issues",  # 공백 문제
                "minor_formatting"  # 사소한 형식 문제
            ]

class QualityAgent:
    """
    교육자료 품질 검수 에이전트
    DocumentAgent가 생성한 슬라이드 데이터를 기반으로 
    실제 학습에 영향을 주는 중요한 품질 이슈만 선별하여 검출
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-5-nano",
        vision_model: str = "gpt-5-nano",
        config: Optional[QualityConfig] = None
    ):
        self.openai_api_key = openai_api_key
        if not openai_api_key:
            raise ValueError("OpenAI API Key가 필요합니다.")
        
        self.model = model
        self.vision_model = vision_model
        self.config = config or QualityConfig()
        
        # OpenAI 클라이언트
        self.client = AsyncOpenAI(api_key=openai_api_key)
        
        # 시스템 프롬프트 설정
        self._setup_system_prompts()
        
        logger.info(f"QualityAgent 초기화 완료")
        logger.info(f"  Vision 모델: {vision_model} (필수 사용)")
        logger.info(f"  슬라이드당 최대 이슈: {self.config.max_issues_per_slide}개")
        logger.info(f"  심각도 필터: {self.config.issue_severity_filter}")

    def _passes_severity_filter(self, severity: str) -> bool:
        """심각도 필터 통과 확인"""
        severity_levels = {"low": 1, "medium": 2, "high": 3}
        filter_level = severity_levels.get(self.config.issue_severity_filter, 2)
        issue_level = severity_levels.get(severity, 2)

        return issue_level >= filter_level
    
    def _setup_system_prompts(self):
        """시스템 프롬프트 설정"""
        self.system_prompt = """당신은 교육자료 품질 검수 전문가입니다.

주어진 슬라이드 이미지와 캡션을 함께 분석하여 실제 학습에 방해가 될 수 있는 중요한 문제만 찾아내세요.

**분석 우선순위:**
1. 슬라이드 이미지 직접 분석 (메인 소스)
2. 캡션 정보 참고 (보조 정보)

**중요한 원칙:**
1. 사소하고 trivial한 문제는 무시하세요
2. 학습자의 이해에 실질적으로 영향을 주는 문제에 집중하세요
3. 한 슬라이드당 최대 3개의 가장 중요한 문제만 선별하세요

**검출 대상 이슈 유형:**
- typo: 이미지 내 텍스트의 명백한 오탈자
- grammar: 이미지 내 텍스트의 문법 오류
- fact: 명백한 사실 오류나 잘못된 정보
- image_quality: 이미지 해상도, 선명도, 가독성 문제
- content_clarity: 내용 전달의 명료성 문제
- layout: 레이아웃이나 디자인으로 인한 이해 방해

**중점 검사 항목:**
1. 이미지 내 텍스트의 오탈자 및 문법 오류
2. 다이어그램, 차트, 표의 정확성
3. 텍스트 가독성 (크기, 대비, 배치)
4. 정보의 논리적 일관성
5. 시각적 요소의 교육적 효과

**무시할 문제들:**
- 단순 띄어쓰기, 문장 부호 누락
- 사소한 표현 차이
- 개인적 선호도 문제
- 극히 경미한 형식 문제

응답은 반드시 JSON 배열 형태로 해주세요:
[
    {
        "issue_type": "typo|grammar|fact|image_quality|content_clarity|layout",
        "original_text": "문제가 있는 원본 텍스트 (이미지에서 발견된)",
        "message": "구체적인 문제점 설명",
        "suggestion": "수정 제안",
        "severity": "high|medium|low",
        "confidence": 0.0-1.0,
        "location": "이미지 내 위치 설명 (예: 제목, 본문, 차트 라벨 등)"
    }
]

문제가 없으면 빈 배열 []을 반환하세요."""
    
    async def analyze_document(self, document_agent, doc_id: str) -> List[Issue]:
        """
        DocumentAgent에서 처리된 문서 전체의 품질 검수
        
        Args:
            document_agent: DocumentAgent 인스턴스
            doc_id: 문서 ID
            
        Returns:
            List[Issue]: 발견된 품질 이슈들
        """
        logger.info(f"품질 검수 시작: {doc_id}")
        
        # DocumentAgent에서 슬라이드 데이터 가져오기
        doc_meta = document_agent.get_document(doc_id)
        slide_data_list = document_agent.get_slide_data(doc_id)
        
        if not doc_meta or not slide_data_list:
            logger.warning(f"문서 데이터가 없습니다: {doc_id}")
            return []
        
        all_issues = []
        
        # 슬라이드별 분석
        for slide_data in slide_data_list:
            try:
                logger.info(f"슬라이드 분석: {slide_data['page_id']}")
                
                slide_issues = await self._analyze_slide(slide_data, doc_meta)
                all_issues.extend(slide_issues)
                
                # API 레이트 제한 고려
                await asyncio.sleep(0.2)
                
            except Exception as e:
                logger.error(f"슬라이드 분석 실패 {slide_data['page_id']}: {str(e)}")
                continue
        
        # 문서 레벨 일관성 검사
        # document_issues = await self._analyze_document_consistency(slide_data_list, doc_meta)
        # all_issues.extend(document_issues)
        
        # 최종 필터링
        filtered_issues = self._filter_issues(all_issues)
        
        logger.info(f"품질 검수 완료: {len(filtered_issues)}/{len(all_issues)}개 이슈 선별")
        return filtered_issues
    
    async def _analyze_slide(self, slide_data: Dict[str, Any], doc_meta: DocumentMeta) -> List[Issue]:
        """단일 슬라이드 품질 분석 - 이미지 + 캡션 통합 분석"""
        
        # 이미지가 필수
        if not slide_data.get("image_base64"):
            logger.warning(f"슬라이드 {slide_data['page_id']}에 이미지가 없습니다")
            return []
        
        try:
            # Vision LLM으로 이미지 + 캡션 통합 분석
            issues_data = await self._analyze_slide_with_vision(slide_data)
            
            # Issue 객체로 변환
            issues = self._convert_to_issues(issues_data, slide_data, doc_meta)
            
            return issues
            
        except Exception as e:
            logger.error(f"슬라이드 분석 실패 {slide_data['page_id']}: {str(e)}")
            return []
    
    async def _analyze_slide_with_vision(self, slide_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Vision 모델을 통한 슬라이드 이미지 + 캡션 통합 분석"""
        
        # 캡션 정보 준비
        caption_info = ""
        if slide_data.get("caption"):
            caption_info += f"\n\n[AI 생성 캡션 (참고용)]\n{slide_data['caption']}"
        
        if slide_data.get("slide_text"):
            caption_info += f"\n\n[슬라이드 원본 텍스트 (참고용)]\n{slide_data['slide_text']}"
        
        analysis_prompt = f"""이 교육자료 슬라이드를 면밀히 분석해주세요.

**분석 방법:**
1. 슬라이드 이미지를 직접 보고 분석하세요 (주요 소스)
2. 아래 캡션 정보는 참고만 하세요 (보조 정보)

**중점 검사 항목:**
1. 이미지 내 모든 텍스트의 오탈자, 문법 오류
2. 차트, 표, 다이어그램의 정확성
3. 텍스트 가독성 (크기, 대비, 위치)
4. 정보의 논리적 일관성
5. 시각적 요소가 학습에 미치는 영향

**특히 주의깊게 확인할 것:**
- 제목, 헤딩의 오탈자
- 본문 텍스트의 문법 오류
- 차트나 표의 라벨, 수치 오류
- 용어 사용의 일관성
- 텍스트와 시각 요소 간의 불일치

위의 시스템 지침에 따라 실제 학습에 방해가 될 수 있는 중요한 문제만 찾아서 JSON 배열로 반환해주세요.{caption_info}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": analysis_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{slide_data['image_base64']}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
            )
            
            response_text = response.choices[0].message.content.strip()

            
            # JSON 파싱
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            issues_data = json.loads(response_text)
            
            # 이슈 개수 제한
            if len(issues_data) > self.config.max_issues_per_slide:
                # 심각도와 신뢰도 기준으로 상위 N개 선택
                issues_data = sorted(
                    issues_data, 
                    key=lambda x: (
                        x.get("severity", "medium") == "high",
                        x.get("confidence", 0.5)
                    ),
                    reverse=True
                )[:self.config.max_issues_per_slide]
            
            return issues_data
            
        except json.JSONDecodeError as e:
            logger.warning(f"Vision LLM 응답 JSON 파싱 실패: {response_text[:100]}... - {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Vision 분석 실패: {str(e)}")
            return []
    
    def _prepare_slide_text(self, slide_data: Dict[str, Any]) -> str:
        """슬라이드 분석용 텍스트 준비"""
        text_parts = []
        
        # 캡션 (주요 분석 대상)
        if slide_data.get("caption"):
            text_parts.append(f"[슬라이드 설명]\n{slide_data['caption']}")
        
        # 원본 슬라이드 텍스트 (있는 경우)
        if slide_data.get("slide_text"):
            text_parts.append(f"[슬라이드 원본 텍스트]\n{slide_data['slide_text']}")
        
        return "\n\n".join(text_parts)
    
    async def _request_llm_analysis(self, text: str) -> List[Dict[str, Any]]:
        """LLM에 품질 분석 요청"""
        
        user_prompt = f"""다음 교육자료 슬라이드를 분석해주세요:

{text}

위의 시스템 지침에 따라 실제 학습에 방해가 될 수 있는 중요한 문제만 찾아서 JSON 배열로 반환해주세요."""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        
        response_text = response.choices[0].message.content.strip()
        
        try:
            # JSON 파싱
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            issues_data = json.loads(response_text)
            
            # 이슈 개수 제한
            if len(issues_data) > self.config.max_issues_per_slide:
                # 심각도와 신뢰도 기준으로 상위 N개 선택
                issues_data = sorted(
                    issues_data, 
                    key=lambda x: (
                        x.get("severity", "medium") == "high",
                        x.get("confidence", 0.5)
                    ),
                    reverse=True
                )[:self.config.max_issues_per_slide]
            
            return issues_data
            
        except json.JSONDecodeError as e:
            logger.warning(f"LLM 응답 JSON 파싱 실패: {response_text[:100]}... - {str(e)}")
            return []
    
    def _convert_to_issues(
        self, 
        issues_data: List[Dict[str, Any]], 
        slide_data: Dict[str, Any], 
        doc_meta: DocumentMeta
    ) -> List[Issue]:
        """Vision LLM 응답을 Issue 객체로 변환"""
        
        issues = []
        
        for issue_data in issues_data:
            try:
                # 필수 필드 확인
                if not all(key in issue_data for key in ["issue_type", "message"]):
                    logger.warning(f"이슈 데이터 필드 누락: {issue_data}")
                    continue
                
                # IssueType 검증 (layout 추가 처리)
                issue_type_str = issue_data["issue_type"]
                if issue_type_str == "layout":
                    issue_type_str = "image_quality"  # 기존 타입으로 매핑
                
                try:
                    issue_type = IssueType(issue_type_str)
                except ValueError:
                    logger.warning(f"지원하지 않는 이슈 타입: {issue_data['issue_type']}")
                    continue
                
                # 신뢰도 필터링
                confidence = issue_data.get("confidence", 0.8)
                if confidence < self.config.confidence_threshold:
                    continue
                
                # 심각도 필터링
                severity = issue_data.get("severity", "medium")
                if not self._passes_severity_filter(severity):
                    continue
                
                # 원본 텍스트와 위치 정보
                original_text = issue_data.get("original_text", "")
                location_desc = issue_data.get("location", "")
                
                # 이미지 기반 분석이므로 text_location은 None으로 설정
                text_location = None
                
                # Issue 객체 생성
                issue = Issue(
                    issue_id=generate_issue_id(
                        doc_meta.doc_id,
                        slide_data["page_id"],
                        TextLocation(start=0, end=1),  # 더미 위치
                        issue_type
                    ),
                    doc_id=doc_meta.doc_id,
                    page_id=slide_data["page_id"],
                    issue_type=issue_type,
                    text_location=text_location,
                    bbox_location=None,
                    element_id=None,
                    original_text=original_text,
                    message=f"[{location_desc}] {issue_data['message']}" if location_desc else issue_data["message"],
                    suggestion=issue_data.get("suggestion", ""),
                    confidence=confidence,
                    confidence_level="high",  # Pydantic이 자동 계산
                    agent_name="quality_agent_vision"
                )
                
                issues.append(issue)
                
            except Exception as e:
                logger.warning(f"이슈 객체 생성 실패: {str(e)} - {issue_data}")
                continue
        
        return issues
    
    def _find_text_location(self, slide_data: Dict[str, Any], original_text: str) -> Optional[TextLocation]:
        """텍스트에서 원본 위치 찾기"""
        if not original_text:
            return None
        
        # 캡션에서 찾기
        caption = slide_data.get("caption", "")
        if original_text in caption:
            start = caption.find(original_text)
            return TextLocation(start=start, end=start + len(original_text))
        
        # 원본 텍스트에서 찾기
        slide_text = slide_data.get("slide_text", "")
        if original_text in slide_text:
            start = slide_text.find(original_text)
            return TextLocation(start=start, end=start + len(original_text))
        
        return None
    
    
#     async def _analyze_document_consistency(self, slide_data_list: List[Dict[str, Any]], doc_meta: DocumentMeta) -> List[Issue]:
#         """문서 전체 일관성 분석 - 캡션 기반"""
#         if len(slide_data_list) < 2:
#             return []
        
#         try:
#             # 모든 캡션 결합 (이미지 분석은 개별 슬라이드에서 수행)
#             all_captions = []
#             for slide in slide_data_list:
#                 if slide.get("caption"):
#                     all_captions.append(f"슬라이드 {slide['page_number']}: {slide['caption']}")
            
#             if len(all_captions) < 2:
#                 return []
            
#             combined_text = "\n\n".join(all_captions)
            
#             consistency_prompt = f"""다음은 교육자료의 모든 슬라이드 AI 생성 캡션입니다:

# {combined_text}

# 문서 전체에서 다음 일관성 문제를 찾아주세요:
# 1. 동일한 개념에 대한 다른 용어 사용 (예: "머신러닝" vs "기계학습")
# 2. 설명 스타일의 심각한 불일치
# 3. 논리적 순서나 구조의 문제
# 4. 전체적인 교육 흐름의 문제

# 중요한 일관성 문제만 JSON 배열로 반환해주세요:
# [{{"issue_type": "consistency", "message": "...", "suggestion": "...", "affected_slides": [1, 2, 3], "confidence": 0.0-1.0}}]

# 문제가 없으면 빈 배열 []을 반환하세요."""

#             response = await self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": "당신은 교육자료 일관성 검수 전문가입니다."},
#                     {"role": "user", "content": consistency_prompt}
#                 ],
#             )
            
#             response_text = response.choices[0].message.content.strip()
            
#             # JSON 파싱
#             if response_text.startswith("```json"):
#                 response_text = response_text.split("```json")[1].split("```")[0].strip()
#             elif response_text.startswith("```"):
#                 response_text = response_text.split("```")[1].split("```")[0].strip()
            
#             consistency_data = json.loads(response_text)
            
#             # 문서 레벨 이슈로 변환
#             document_issues = []
#             for issue_data in consistency_data:
#                 if issue_data.get("confidence", 0.8) >= self.config.confidence_threshold:
#                     # 첫 번째 슬라이드에 이슈 할당
#                     first_slide = slide_data_list[0]
                    
#                     issue = Issue(
#                         issue_id=generate_issue_id(
#                             doc_meta.doc_id,
#                             first_slide["page_id"],
#                             TextLocation(start=0, end=1),
#                             IssueType.CONSISTENCY
#                         ),
#                         doc_id=doc_meta.doc_id,
#                         page_id=first_slide["page_id"],
#                         issue_type=IssueType.CONSISTENCY,
#                         text_location=None,
#                         bbox_location=None,
#                         element_id=None,
#                         original_text="문서 전체",
#                         message=f"[문서 일관성] {issue_data['message']}",
#                         suggestion=issue_data.get("suggestion", ""),
#                         confidence=issue_data.get("confidence", 0.8),
#                         confidence_level="medium",
#                         agent_name="quality_agent_consistency"
#                     )
                    
#                     document_issues.append(issue)
            
#             return document_issues
            
#         except Exception as e:
#             logger.warning(f"문서 일관성 분석 실패: {str(e)}")
#             return []
    
    def _filter_issues(self, issues: List[Issue]) -> List[Issue]:
        """최종 이슈 필터링"""
        filtered = []
        
        for issue in issues:
            # 신뢰도 필터
            if issue.confidence < self.config.confidence_threshold:
                continue
            
            # 중복 제거 (같은 페이지, 같은 타입, 비슷한 메시지)
            is_duplicate = False
            for existing in filtered:
                if (existing.page_id == issue.page_id and 
                    existing.issue_type == issue.issue_type and
                    self._similar_messages(existing.message, issue.message)):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(issue)
        
        # 심각도 기준 정렬
        filtered.sort(key=lambda x: (
            x.issue_type == IssueType.FACT,  # 사실 오류 우선
            x.confidence
        ), reverse=True)
        
        return filtered
    
    def _similar_messages(self, msg1: str, msg2: str, threshold: float = 0.8) -> bool:
        """메시지 유사도 확인 (간단한 문자열 비교)"""
        if not msg1 or not msg2:
            return False
        
        # 간단한 Jaccard 유사도
        words1 = set(msg1.lower().split())
        words2 = set(msg2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return (intersection / union) > threshold
    
    def get_quality_summary(self, issues: List[Issue]) -> Dict[str, Any]:
        """품질 검수 결과 요약"""
        if not issues:
            return {
                "total_issues": 0,
                "quality_score": 1.0,
                "by_type": {},
                "by_severity": {},
                "recommendations": ["문서 품질이 우수합니다."]
            }
        
        # 타입별 분류
        by_type = {}
        by_severity = {"high": 0, "medium": 0, "low": 0}
        
        for issue in issues:
            issue_type = issue.issue_type.value
            by_type[issue_type] = by_type.get(issue_type, 0) + 1
            
            if issue.confidence >= 0.9:
                by_severity["high"] += 1
            elif issue.confidence >= 0.7:
                by_severity["medium"] += 1
            else:
                by_severity["low"] += 1
        
        # 품질 점수 계산 (0.0 ~ 1.0)
        quality_score = max(0.0, 1.0 - (len(issues) * 0.1))
        
        # 권장사항 생성
        recommendations = []
        if by_type.get("fact", 0) > 0:
            recommendations.append("사실 확인이 필요한 내용이 있습니다.")
        if by_type.get("typo", 0) > 0:
            recommendations.append("오탈자 교정이 필요합니다.")
        if by_type.get("consistency", 0) > 0:
            recommendations.append("용어 사용의 일관성을 확인해주세요.")
        if by_type.get("image_quality", 0) > 0:
            recommendations.append("이미지 품질 개선을 고려해주세요.")
        
        return {
            "total_issues": len(issues),
            "quality_score": quality_score,
            "by_type": by_type,
            "by_severity": by_severity,
            "recommendations": recommendations
        }

async def test_quality_agent_e2e():
    """QualityAgent E2E 테스트 - DocumentAgent와 연동"""
    print("🧪 QualityAgent E2E 테스트 시작...")
    
    import os
    from dotenv import load_dotenv
    from pathlib import Path
    
    env_path = Path(__file__).resolve().parents[2] / '.env.dev'
    load_dotenv(env_path)
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        return
    
    # 테스트할 파일 찾기
    test_files = [
        "sample_docs/sample.pdf"
    ]
    
    test_file = None
    for file_name in test_files:
        if Path(file_name).exists():
            test_file = file_name
            break
    
    if not test_file:
        print("❌ 테스트할 PDF 파일이 없습니다.")
        print(f"   다음 파일 중 하나를 준비해주세요: {', '.join(test_files)}")
        return
    
    print(f"📄 테스트 파일: {test_file}")
    
    try:
        # 1. DocumentAgent 초기화 및 문서 처리
        print("\n🔧 DocumentAgent 초기화 중...")
        from src.agents.document_agent import DocumentAgent  # 실제 임포트 경로에 맞게 수정
        
        document_agent = DocumentAgent(
            openai_api_key=api_key,
            vision_model="gpt-5-nano",
            embedding_model="text-embedding-3-small"
        )
        
        print(f"📖 문서 처리 중: {test_file}")
        doc_meta = await document_agent.process_document(test_file)
        
        print(f"✅ 문서 처리 완료!")
        print(f"   문서 ID: {doc_meta.doc_id}")
        print(f"   제목: {doc_meta.title}")
        print(f"   슬라이드 수: {doc_meta.total_pages}")
        
        # 2. 슬라이드 데이터 확인
        slide_data_list = document_agent.get_slide_data(doc_meta.doc_id)
        print(f"   생성된 슬라이드 데이터: {len(slide_data_list)}개")
        
        # 첫 번째 슬라이드 정보 출력
        if slide_data_list:
            first_slide = slide_data_list[0]
            print(f"   첫 슬라이드 캡션: {first_slide.get('caption', 'None')[:100]}...")
            print(f"   이미지 데이터: {'있음' if first_slide.get('image_base64') else '없음'}")
        
        # 3. QualityAgent 초기화 및 품질 검수
        print("\n🔍 QualityAgent 초기화 중...")
        config = QualityConfig(
            max_issues_per_slide=3,
            confidence_threshold=0.7,
            issue_severity_filter="medium"
        )
        
        quality_agent = QualityAgent(
            openai_api_key=api_key,
            vision_model="gpt-5-nano",
            config=config
        )
        
        print("📋 품질 검수 실행 중... (Vision 모델 사용)")
        print("   ⏳ 이 과정은 몇 분이 소요될 수 있습니다.")
        
        issues = await quality_agent.analyze_document(document_agent, doc_meta.doc_id)
        
        # 4. 결과 출력
        print(f"\n📋 발견된 이슈들 ({len(issues)}개):")
        
        if not issues:
            print("   🎉 발견된 품질 이슈가 없습니다!")
        else:
            # 이슈 출력
            for i, issue in enumerate(issues, 1):
                print(f"\n{i}. [{issue.issue_type.value.upper()}] {issue.page_id}")
                print(f"   원본: {issue.original_text[:50]}{'...' if len(issue.original_text) > 50 else ''}")
                print(f"   문제: {issue.message}")
                print(f"   제안: {issue.suggestion}")
                print(f"   신뢰도: {issue.confidence:.2f}")
                print(f"   에이전트: {issue.agent_name}")
        
        # 5. 품질 요약
        summary = quality_agent.get_quality_summary(issues)
        print(f"\n📊 품질 요약:")
        print(f"   전체 이슈: {summary['total_issues']}개")
        print(f"   품질 점수: {summary['quality_score']:.2f}/1.0")
        
        if summary['by_type']:
            print(f"\n📈 이슈 유형별 분포:")
            for issue_type, count in summary['by_type'].items():
                print(f"   {issue_type}: {count}개")
        
        if summary['by_severity']:
            print(f"\n⚖️ 심각도별 분포:")
            for severity, count in summary['by_severity'].items():
                print(f"   {severity}: {count}개")
        
        print(f"\n🎯 권장사항:")
        for rec in summary['recommendations']:
            print(f"   - {rec}")
        
        # 6. 추가 정보
        stats = document_agent.get_document_stats(doc_meta.doc_id)
        print(f"\n📈 문서 통계:")
        print(f"   캡션 생성률: {stats['caption_coverage']:.1%}")
        print(f"   평균 캡션 길이: {stats['avg_caption_length']:.0f}자")
        print(f"   총 이미지 크기: {stats['total_image_size_mb']:.1f}MB")
        
        print("\n🎉 E2E 테스트 완료!")
        
    except ImportError as e:
        print(f"❌ 모듈 임포트 실패: {str(e)}")
        print("   DocumentAgent 클래스의 임포트 경로를 확인해주세요.")
        
    except FileNotFoundError as e:
        print(f"❌ 파일 오류: {str(e)}")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_quality_agent_e2e())