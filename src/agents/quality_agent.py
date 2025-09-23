"""
EDU-Audit Multimodal Quality Agent
오탈자, 문법, 표현 일관성 + 멀티모달 품질 검출 에이전트
"""

import asyncio
import logging
import re
from collections import defaultdict, Counter
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass

from llama_index.llms.openai import OpenAI

from src.core.models import (
    DocumentMeta, PageInfo, PageElement, ElementType, Issue, IssueType, 
    TextLocation, BoundingBox, ImageElement, TableElement, ChartElement,
    generate_issue_id
)

logger = logging.getLogger(__name__)


@dataclass
class TypoPattern:
    """오탈자 패턴 정의"""
    pattern: str          # 정규식 패턴
    correction: str       # 올바른 표현
    confidence: float     # 신뢰도
    description: str      # 설명


@dataclass
class ConsistencyRule:
    """일관성 규칙 정의"""
    terms: List[str]      # 동일 의미의 다른 표현들
    preferred: str        # 권장 표현
    description: str      # 설명
    domain: str          # 적용 도메인


@dataclass
class ImageQualityRule:
    """이미지 품질 규칙"""
    min_width: int = 300
    min_height: int = 200
    max_file_size_mb: float = 5.0
    required_dpi: int = 150
    avoid_formats: List[str] = None
    
    def __post_init__(self):
        if self.avoid_formats is None:
            self.avoid_formats = ["bmp", "tiff"]


@dataclass
class TableQualityRule:
    """표 품질 규칙"""
    min_rows: int = 2
    min_cols: int = 2
    max_empty_cells_ratio: float = 0.3
    require_headers: bool = True
    consistent_column_count: bool = True


@dataclass
class ChartQualityRule:
    """차트 품질 규칙"""
    require_title: bool = True
    require_axis_labels: bool = True
    require_legend: bool = False
    min_data_points: int = 2
    readable_text_size: bool = True


class MultimodalQualityAgent:
    """멀티모달 문서 품질 검사 에이전트"""
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None, 
        llm_model: str = "gpt-5-nano",
        vision_model: str = "gpt-4-vision-preview",
        enable_vision_analysis: bool = True
    ):
        self.openai_api_key = openai_api_key
        self.enable_vision_analysis = enable_vision_analysis and openai_api_key
        
        # LLM 초기화 (API 키가 있는 경우만)
        self.llm = None
        self.vision_llm = None
        
        if openai_api_key:
            self.llm = OpenAI(
                model=llm_model,
                temperature=0.1,
                api_key=openai_api_key
            )
            
            if self.enable_vision_analysis:
                self.vision_llm = OpenAI(
                    model=vision_model,
                    temperature=0.1,
                    api_key=openai_api_key
                )
        
        # 기존 패턴과 규칙 로드
        self._load_typo_patterns()
        self._load_consistency_rules()
        
        # 멀티모달 품질 규칙 로드
        self._load_multimodal_rules()
        
        logger.info("MultimodalQualityAgent 초기화 완료")
        logger.info(f"  Vision 분석: {'활성화' if self.enable_vision_analysis else '비활성화'}")
    
    def _load_typo_patterns(self):
        """일반적인 오탈자 패턴 로드"""
        self.typo_patterns = [
            # 한국어 오탈자
            TypoPattern(
                pattern=r"알고리듬",
                correction="알고리즘",
                confidence=0.95,
                description="알고리듬 → 알고리즘"
            ),
            TypoPattern(
                pattern=r"데이타",
                correction="데이터",
                confidence=0.98,
                description="데이타 → 데이터"
            ),
            TypoPattern(
                pattern=r"컴퓨타",
                correction="컴퓨터",
                confidence=0.98,
                description="컴퓨타 → 컴퓨터"
            ),
            TypoPattern(
                pattern=r"앨고리즘",
                correction="알고리즘",
                confidence=0.90,
                description="앨고리즘 → 알고리즘"
            ),
            
            # 영어 오탈자
            TypoPattern(
                pattern=r"\bteh\b",
                correction="the",
                confidence=0.99,
                description="teh → the"
            ),
            TypoPattern(
                pattern=r"\baccomodate\b",
                correction="accommodate",
                confidence=0.95,
                description="accomodate → accommodate"
            ),
            TypoPattern(
                pattern=r"\brecieve\b",
                correction="receive",
                confidence=0.95,
                description="recieve → receive"
            ),
        ]
    
    def _load_consistency_rules(self):
        """용어 일관성 규칙 로드"""
        self.consistency_rules = [
            ConsistencyRule(
                terms=["학습률", "러닝레이트", "러닝 레이트", "학습속도", "learning rate"],
                preferred="학습률",
                description="학습률 관련 용어 통일",
                domain="machine_learning"
            ),
            ConsistencyRule(
                terms=["딥러닝", "심층학습", "깊은학습", "deep learning"],
                preferred="딥러닝",
                description="딥러닝 관련 용어 통일",
                domain="machine_learning"
            ),
            ConsistencyRule(
                terms=["데이터셋", "데이터세트", "데이터 셋", "dataset"],
                preferred="데이터셋",
                description="데이터셋 관련 용어 통일",
                domain="data_science"
            ),
            ConsistencyRule(
                terms=["알고리즘", "알고리듬", "앨고리즘", "algorithm"],
                preferred="알고리즘",
                description="알고리즘 관련 용어 통일",
                domain="computer_science"
            ),
            ConsistencyRule(
                terms=["머신러닝", "기계학습", "머신 러닝", "machine learning"],
                preferred="머신러닝",
                description="머신러닝 관련 용어 통일",
                domain="machine_learning"
            ),
        ]
    
    def _load_multimodal_rules(self):
        """멀티모달 품질 규칙 로드"""
        self.image_quality_rules = ImageQualityRule()
        self.table_quality_rules = TableQualityRule()
        self.chart_quality_rules = ChartQualityRule()
    
    async def check_document(self, doc_meta: DocumentMeta, pages: List[PageInfo]) -> List[Issue]:
        """
        멀티모달 문서 전체 품질 검사
        
        Args:
            doc_meta: 문서 메타데이터
            pages: 페이지 목록 (멀티모달 요소 포함)
            
        Returns:
            List[Issue]: 발견된 이슈들
        """
        logger.info(f"멀티모달 문서 품질 검사 시작: {doc_meta.doc_id}")
        
        all_issues = []
        
        # 1. 기존 텍스트 기반 검사
        text_issues = await self._check_text_quality(doc_meta, pages)
        all_issues.extend(text_issues)
        
        # 2. 이미지 품질 검사
        image_issues = await self._check_image_quality(doc_meta, pages)
        all_issues.extend(image_issues)
        
        # 3. 표 품질 검사
        table_issues = await self._check_table_quality(doc_meta, pages)
        all_issues.extend(table_issues)
        
        # 4. 차트 품질 검사
        chart_issues = await self._check_chart_quality(doc_meta, pages)
        all_issues.extend(chart_issues)
        
        # 5. 멀티모달 일관성 검사 (OCR 텍스트 포함)
        multimodal_consistency_issues = await self._check_multimodal_consistency(doc_meta, pages)
        all_issues.extend(multimodal_consistency_issues)
        
        logger.info(f"멀티모달 품질 검사 완료: {len(all_issues)}개 이슈 발견")
        return all_issues
    
    async def _check_text_quality(self, doc_meta: DocumentMeta, pages: List[PageInfo]) -> List[Issue]:
        """기존 텍스트 품질 검사"""
        all_issues = []
        
        # 1. 패턴 기반 오탈자 검사
        typo_issues = await self._check_typos_pattern_based(doc_meta, pages)
        all_issues.extend(typo_issues)
        
        # 2. 용어 일관성 검사
        consistency_issues = await self._check_consistency(doc_meta, pages)
        all_issues.extend(consistency_issues)
        
        # 3. LLM 기반 문법 검사
        if self.llm:
            grammar_issues = await self._check_grammar_llm(doc_meta, pages)
            all_issues.extend(grammar_issues)
        
        return all_issues
    
    async def _check_image_quality(self, doc_meta: DocumentMeta, pages: List[PageInfo]) -> List[Issue]:
        """이미지 품질 검사"""
        issues = []
        
        for page in pages:
            for element in page.elements:
                if element.element_type != ElementType.IMAGE or not element.image_data:
                    continue
                
                image_data = element.image_data
                
                # 1. 이미지 크기 검사
                if (image_data.dimensions[0] < self.image_quality_rules.min_width or 
                    image_data.dimensions[1] < self.image_quality_rules.min_height):
                    
                    issue = self._create_multimodal_issue(
                        doc_meta, page, element,
                        IssueType.IMAGE_QUALITY,
                        f"이미지 해상도가 낮습니다 ({image_data.dimensions[0]}x{image_data.dimensions[1]})",
                        f"최소 {self.image_quality_rules.min_width}x{self.image_quality_rules.min_height} 권장",
                        0.8
                    )
                    issues.append(issue)
                
                # 2. 파일 크기 검사
                size_mb = image_data.size_bytes / (1024 * 1024)
                if size_mb > self.image_quality_rules.max_file_size_mb:
                    issue = self._create_multimodal_issue(
                        doc_meta, page, element,
                        IssueType.IMAGE_QUALITY,
                        f"이미지 파일 크기가 큽니다 ({size_mb:.1f}MB)",
                        f"최대 {self.image_quality_rules.max_file_size_mb}MB 권장",
                        0.7
                    )
                    issues.append(issue)
                
                # 3. 파일 형식 검사
                if image_data.format.lower() in self.image_quality_rules.avoid_formats:
                    issue = self._create_multimodal_issue(
                        doc_meta, page, element,
                        IssueType.IMAGE_QUALITY,
                        f"권장하지 않는 이미지 형식입니다 ({image_data.format})",
                        "JPG, PNG 형식 사용 권장",
                        0.6
                    )
                    issues.append(issue)
                
                # 4. Vision LLM을 통한 품질 분석
                if self.enable_vision_analysis and self.vision_llm:
                    vision_issues = await self._analyze_image_quality_with_vision(
                        doc_meta, page, element
                    )
                    issues.extend(vision_issues)
        
        logger.info(f"이미지 품질 검사: {len(issues)}개 발견")
        return issues
    
    async def _check_table_quality(self, doc_meta: DocumentMeta, pages: List[PageInfo]) -> List[Issue]:
        """표 품질 검사"""
        issues = []
        
        for page in pages:
            for element in page.elements:
                if element.element_type != ElementType.TABLE or not element.table_data:
                    continue
                
                table_data = element.table_data
                
                # 1. 최소 크기 검사
                if (table_data.row_count < self.table_quality_rules.min_rows or 
                    table_data.col_count < self.table_quality_rules.min_cols):
                    
                    issue = self._create_multimodal_issue(
                        doc_meta, page, element,
                        IssueType.TABLE_FORMAT,
                        f"표 크기가 작습니다 ({table_data.row_count}x{table_data.col_count})",
                        f"최소 {self.table_quality_rules.min_rows}x{self.table_quality_rules.min_cols} 권장",
                        0.7
                    )
                    issues.append(issue)
                
                # 2. 헤더 존재 여부 검사
                if self.table_quality_rules.require_headers and not table_data.has_headers:
                    issue = self._create_multimodal_issue(
                        doc_meta, page, element,
                        IssueType.TABLE_FORMAT,
                        "표에 헤더가 없습니다",
                        "명확한 열 제목 추가 권장",
                        0.8
                    )
                    issues.append(issue)
                
                # 3. 빈 셀 비율 검사
                empty_cells = self._count_empty_cells(table_data)
                total_cells = table_data.row_count * table_data.col_count
                empty_ratio = empty_cells / total_cells if total_cells > 0 else 0
                
                if empty_ratio > self.table_quality_rules.max_empty_cells_ratio:
                    issue = self._create_multimodal_issue(
                        doc_meta, page, element,
                        IssueType.TABLE_FORMAT,
                        f"표에 빈 셀이 많습니다 ({empty_ratio:.1%})",
                        "불필요한 빈 셀 제거 또는 데이터 보완 권장",
                        0.6
                    )
                    issues.append(issue)
                
                # 4. 열 일관성 검사
                if self.table_quality_rules.consistent_column_count:
                    inconsistent_rows = self._check_column_consistency(table_data)
                    if inconsistent_rows:
                        issue = self._create_multimodal_issue(
                            doc_meta, page, element,
                            IssueType.TABLE_FORMAT,
                            f"표의 열 개수가 일관되지 않습니다 ({len(inconsistent_rows)}개 행)",
                            "모든 행의 열 개수 통일 권장",
                            0.9
                        )
                        issues.append(issue)
        
        logger.info(f"표 품질 검사: {len(issues)}개 발견")
        return issues
    
    async def _check_chart_quality(self, doc_meta: DocumentMeta, pages: List[PageInfo]) -> List[Issue]:
        """차트 품질 검사"""
        issues = []
        
        for page in pages:
            for element in page.elements:
                if element.element_type != ElementType.CHART or not element.chart_data:
                    continue
                
                chart_data = element.chart_data
                
                # 1. 제목 존재 여부 검사
                if self.chart_quality_rules.require_title and not chart_data.title:
                    issue = self._create_multimodal_issue(
                        doc_meta, page, element,
                        IssueType.CHART_READABILITY,
                        "차트에 제목이 없습니다",
                        "차트 내용을 설명하는 제목 추가 권장",
                        0.8
                    )
                    issues.append(issue)
                
                # 2. 축 라벨 검사
                if (self.chart_quality_rules.require_axis_labels and 
                    (not chart_data.x_label or not chart_data.y_label)):
                    
                    missing_labels = []
                    if not chart_data.x_label:
                        missing_labels.append("X축")
                    if not chart_data.y_label:
                        missing_labels.append("Y축")
                    
                    issue = self._create_multimodal_issue(
                        doc_meta, page, element,
                        IssueType.CHART_READABILITY,
                        f"차트에 {', '.join(missing_labels)} 라벨이 없습니다",
                        "축의 의미를 설명하는 라벨 추가 권장",
                        0.7
                    )
                    issues.append(issue)
                
                # 3. 범례 검사
                if (self.chart_quality_rules.require_legend and 
                    not chart_data.legend):
                    
                    issue = self._create_multimodal_issue(
                        doc_meta, page, element,
                        IssueType.CHART_READABILITY,
                        "차트에 범례가 없습니다",
                        "데이터 시리즈 구분을 위한 범례 추가 권장",
                        0.6
                    )
                    issues.append(issue)
                
                # 4. 데이터 포인트 수 검사
                if len(chart_data.data_points) < self.chart_quality_rules.min_data_points:
                    issue = self._create_multimodal_issue(
                        doc_meta, page, element,
                        IssueType.CHART_READABILITY,
                        f"차트 데이터가 부족합니다 ({len(chart_data.data_points)}개)",
                        f"최소 {self.chart_quality_rules.min_data_points}개 데이터 포인트 권장",
                        0.7
                    )
                    issues.append(issue)
                
                # 5. Vision LLM을 통한 가독성 분석
                if self.enable_vision_analysis and self.vision_llm:
                    readability_issues = await self._analyze_chart_readability_with_vision(
                        doc_meta, page, element
                    )
                    issues.extend(readability_issues)
        
        logger.info(f"차트 품질 검사: {len(issues)}개 발견")
        return issues
    
    async def _check_multimodal_consistency(self, doc_meta: DocumentMeta, pages: List[PageInfo]) -> List[Issue]:
        """멀티모달 요소간 일관성 검사 (OCR 텍스트 포함)"""
        issues = []
        
        # 모든 텍스트 수집 (raw_text + OCR 텍스트)
        all_texts = []
        
        for page in pages:
            # 페이지 텍스트
            if page.raw_text.strip():
                all_texts.append(page.raw_text)
            
            # OCR 텍스트
            for element in page.elements:
                if (element.element_type == ElementType.IMAGE and 
                    element.image_data and 
                    element.image_data.ocr_text):
                    all_texts.append(element.image_data.ocr_text)
        
        # 전체 텍스트에서 일관성 검사
        full_text = "\n".join(all_texts)
        
        for rule in self.consistency_rules:
            found_terms = self._find_terms_in_text(full_text, rule.terms)
            
            if len(found_terms) > 1:
                # OCR 텍스트에서 발견된 불일치 용어에 대한 이슈 생성
                for term, positions in found_terms.items():
                    if term != rule.preferred:
                        # OCR 텍스트에서 발견된 경우 특별 처리
                        ocr_issue = self._find_ocr_inconsistency(
                            doc_meta, pages, term, rule
                        )
                        if ocr_issue:
                            issues.append(ocr_issue)
        
        logger.info(f"멀티모달 일관성 검사: {len(issues)}개 발견")
        return issues
    
    def _create_multimodal_issue(
        self, 
        doc_meta: DocumentMeta, 
        page: PageInfo, 
        element: PageElement,
        issue_type: IssueType,
        message: str,
        suggestion: str,
        confidence: float
    ) -> Issue:
        """멀티모달 이슈 생성"""
        
        issue_id = generate_issue_id(
            doc_meta.doc_id,
            page.page_id,
            element.bbox or BoundingBox(x=0, y=0, width=100, height=100),
            issue_type
        )
        
        return Issue(
            issue_id=issue_id,
            doc_id=doc_meta.doc_id,
            page_id=page.page_id,
            issue_type=issue_type,
            text_location=None,  # 멀티모달 요소는 bbox 사용
            bbox_location=element.bbox,
            element_id=element.element_id,
            original_text=self._get_element_description(element),
            message=message,
            suggestion=suggestion,
            confidence=confidence,
            confidence_level="high",  # Pydantic이 자동 계산
            agent_name="multimodal_quality_agent"
        )
    
    def _get_element_description(self, element: PageElement) -> str:
        """요소 설명 텍스트 생성"""
        if element.element_type == ElementType.IMAGE and element.image_data:
            return element.image_data.description or f"이미지 ({element.image_data.format})"
        elif element.element_type == ElementType.TABLE and element.table_data:
            return f"표 ({element.table_data.row_count}x{element.table_data.col_count})"
        elif element.element_type == ElementType.CHART and element.chart_data:
            return f"{element.chart_data.chart_type} 차트"
        else:
            return f"{element.element_type.value} 요소"
    
    def _count_empty_cells(self, table_data: TableElement) -> int:
        """표의 빈 셀 개수 계산"""
        empty_count = 0
        
        for row in table_data.rows:
            for cell in row:
                if not cell or cell.strip() == "":
                    empty_count += 1
        
        return empty_count
    
    def _check_column_consistency(self, table_data: TableElement) -> List[int]:
        """표의 열 일관성 검사"""
        inconsistent_rows = []
        expected_cols = len(table_data.headers) if table_data.headers else None
        
        for i, row in enumerate(table_data.rows):
            if expected_cols is None:
                expected_cols = len(row)
            elif len(row) != expected_cols:
                inconsistent_rows.append(i)
        
        return inconsistent_rows
    
    def _find_ocr_inconsistency(
        self, 
        doc_meta: DocumentMeta, 
        pages: List[PageInfo], 
        term: str, 
        rule: ConsistencyRule
    ) -> Optional[Issue]:
        """OCR 텍스트에서 용어 불일치 찾기"""
        
        for page in pages:
            for element in page.elements:
                if (element.element_type == ElementType.IMAGE and 
                    element.image_data and 
                    element.image_data.ocr_text and 
                    term.lower() in element.image_data.ocr_text.lower()):
                    
                    return self._create_multimodal_issue(
                        doc_meta, page, element,
                        IssueType.CONSISTENCY,
                        f"이미지 내 텍스트에서 용어 불일치: '{term}'",
                        f"'{rule.preferred}' 용어로 통일 권장",
                        0.8
                    )
        
        return None
    
    async def _analyze_image_quality_with_vision(
        self, 
        doc_meta: DocumentMeta, 
        page: PageInfo, 
        element: PageElement
    ) -> List[Issue]:
        """Vision LLM을 통한 이미지 품질 분석"""
        if not self.vision_llm or not element.image_data or not element.image_data.image_data:
            return []
        
        try:
            prompt = """이 교육용 이미지의 품질을 분석해주세요.

다음 관점에서 문제점이 있는지 확인해주세요:
1. 텍스트 가독성 (흐릿함, 작은 글씨)
2. 이미지 선명도 (픽셀화, 압축 품질)
3. 색상 대비 (구분하기 어려운 색상)
4. 전체적인 시각적 품질

문제가 있으면 "문제발견: [구체적 문제]"로 시작해서 설명해주세요.
문제가 없으면 "품질양호"라고 답해주세요."""

            # 실제 Vision API 호출은 별도 구현 필요
            response = await self._call_vision_api_for_quality(prompt, element.image_data.image_data)
            
            if response and "문제발견:" in response:
                issue = self._create_multimodal_issue(
                    doc_meta, page, element,
                    IssueType.IMAGE_QUALITY,
                    f"Vision 분석: {response.split('문제발견:')[1].strip()}",
                    "이미지 품질 개선 권장",
                    0.7
                )
                return [issue]
        
        except Exception as e:
            logger.warning(f"Vision 기반 이미지 품질 분석 실패: {str(e)}")
        
        return []
    
    async def _analyze_chart_readability_with_vision(
        self, 
        doc_meta: DocumentMeta, 
        page: PageInfo, 
        element: PageElement
    ) -> List[Issue]:
        """Vision LLM을 통한 차트 가독성 분석"""
        # 차트는 보통 이미지 형태로도 존재하므로, 관련 이미지 요소 찾기
        if not self.vision_llm:
            return []
        
        # 차트와 관련된 이미지 요소 찾기 (같은 위치 또는 유사한 설명)
        related_image = None
        for other_element in [e for e in element.__dict__.get('page_elements', [])]:
            if (other_element.element_type == ElementType.IMAGE and 
                other_element.image_data and 
                "차트" in (other_element.image_data.description or "")):
                related_image = other_element
                break
        
        if not related_image or not related_image.image_data.image_data:
            return []
        
        try:
            prompt = """이 차트의 가독성을 분석해주세요.

다음 관점에서 문제점을 확인해주세요:
1. 텍스트 크기 (너무 작거나 읽기 어려움)
2. 색상 구분 (비슷한 색상으로 구분 어려움)
3. 축 라벨과 제목의 명확성
4. 범례의 위치와 가독성
5. 전체적인 레이아웃

문제가 있으면 "가독성문제: [구체적 문제]"로 시작해서 설명해주세요.
문제가 없으면 "가독성양호"라고 답해주세요."""

            response = await self._call_vision_api_for_quality(prompt, related_image.image_data.image_data)
            
            if response and "가독성문제:" in response:
                issue = self._create_multimodal_issue(
                    doc_meta, page, element,
                    IssueType.CHART_READABILITY,
                    f"Vision 분석: {response.split('가독성문제:')[1].strip()}",
                    "차트 가독성 개선 권장",
                    0.7
                )
                return [issue]
        
        except Exception as e:
            logger.warning(f"Vision 기반 차트 가독성 분석 실패: {str(e)}")
        
        return []
    
    async def _call_vision_api_for_quality(self, prompt: str, image_data: str) -> Optional[str]:
        """Vision API 호출 (품질 분석용)"""
        # 실제 구현에서는 OpenAI Vision API 호출
        # 현재는 더미 응답
        await asyncio.sleep(0.1)
        return "품질양호"  # 또는 "문제발견: 텍스트가 흐릿함"
    
    # 기존 텍스트 품질 검사 메서드들 (그대로 유지)
    async def _check_typos_pattern_based(self, doc_meta: DocumentMeta, pages: List[PageInfo]) -> List[Issue]:
        """패턴 기반 오탈자 검사 - 멀티모달 텍스트 포함"""
        issues = []
        
        for page in pages:
            # 1. 페이지 raw_text 검사
            text_issues = await self._check_text_patterns(doc_meta, page, page.raw_text, page.raw_text)
            issues.extend(text_issues)
            
            # 2. OCR 텍스트 검사
            for element in page.elements:
                if (element.element_type == ElementType.IMAGE and 
                    element.image_data and 
                    element.image_data.ocr_text):
                    
                    ocr_issues = await self._check_text_patterns(
                        doc_meta, page, element.image_data.ocr_text, 
                        f"이미지 OCR: {element.image_data.ocr_text}"
                    )
                    # OCR 이슈는 element_id 포함
                    for issue in ocr_issues:
                        issue.element_id = element.element_id
                        issue.message = f"[OCR] {issue.message}"
                    issues.extend(ocr_issues)
        
        logger.info(f"패턴 기반 오탈자 (멀티모달): {len(issues)}개 발견")
        return issues
    
    async def _check_text_patterns(self, doc_meta: DocumentMeta, page: PageInfo, text: str, display_text: str) -> List[Issue]:
        """텍스트에서 패턴 기반 오탈자 검사"""
        issues = []
        
        if not text.strip():
            return issues
        
        for pattern in self.typo_patterns:
            matches = re.finditer(pattern.pattern, text, re.IGNORECASE)
            
            for match in matches:
                location = TextLocation(
                    start=match.start(),
                    end=match.end()
                )
                
                issue_id = generate_issue_id(
                    doc_meta.doc_id,
                    page.page_id,
                    location,
                    IssueType.TYPO
                )
                
                issue = Issue(
                    issue_id=issue_id,
                    doc_id=doc_meta.doc_id,
                    page_id=page.page_id,
                    issue_type=IssueType.TYPO,
                    text_location=location,
                    original_text=match.group(),
                    message=pattern.description,
                    suggestion=pattern.correction,
                    confidence=pattern.confidence,
                    confidence_level="high",
                    agent_name="multimodal_quality_agent"
                )
                
                issues.append(issue)
        
        return issues
    
    async def _check_consistency(self, doc_meta: DocumentMeta, pages: List[PageInfo]) -> List[Issue]:
        """용어 일관성 검사 - 멀티모달 텍스트 포함"""
        issues = []
        
        # 모든 텍스트 수집 (페이지 텍스트 + OCR 텍스트)
        all_texts = []
        for page in pages:
            all_texts.append(page.raw_text)
            
            # OCR 텍스트도 포함
            for element in page.elements:
                if (element.element_type == ElementType.IMAGE and 
                    element.image_data and 
                    element.image_data.ocr_text):
                    all_texts.append(element.image_data.ocr_text)
        
        full_text = "\n".join(all_texts)
        
        for rule in self.consistency_rules:
            found_terms = self._find_terms_in_text(full_text, rule.terms)
            
            if len(found_terms) > 1:
                most_common = max(found_terms.items(), key=lambda x: len(x[1]))[0]
                
                for term, positions in found_terms.items():
                    if term != rule.preferred and term != most_common:
                        first_pos = positions[0]
                        page_info = self._find_page_by_position(pages, first_pos)
                        
                        if page_info:
                            page, relative_pos = page_info
                            
                            location = TextLocation(
                                start=relative_pos,
                                end=relative_pos + len(term)
                            )
                            
                            issue_id = generate_issue_id(
                                doc_meta.doc_id,
                                page.page_id,
                                location,
                                IssueType.CONSISTENCY
                            )
                            
                            issue = Issue(
                                issue_id=issue_id,
                                doc_id=doc_meta.doc_id,
                                page_id=page.page_id,
                                issue_type=IssueType.CONSISTENCY,
                                text_location=location,
                                original_text=term,
                                message=f"용어 일관성: {rule.description}",
                                suggestion=f"'{rule.preferred}' 용어로 통일 권장",
                                confidence=0.85,
                                confidence_level="medium",
                                agent_name="multimodal_quality_agent"
                            )
                            
                            issues.append(issue)
        
        logger.info(f"일관성 검사 (멀티모달): {len(issues)}개 발견")
        return issues
    
    def _find_terms_in_text(self, text: str, terms: List[str]) -> Dict[str, List[int]]:
        """텍스트에서 용어들의 위치 찾기"""
        found = {}
        text_lower = text.lower()
        
        for term in terms:
            positions = []
            term_lower = term.lower()
            start = 0
            
            while True:
                pos = text_lower.find(term_lower, start)
                if pos == -1:
                    break
                
                if self._is_word_boundary(text, pos, len(term)):
                    positions.append(pos)
                
                start = pos + 1
            
            if positions:
                found[term] = positions
        
        return found
    
    def _is_word_boundary(self, text: str, pos: int, length: int) -> bool:
        """단어 경계인지 확인"""
        before = text[pos-1] if pos > 0 else ' '
        after = text[pos + length] if pos + length < len(text) else ' '
        
        return not (before.isalnum() or after.isalnum())
    
    def _find_page_by_position(self, pages: List[PageInfo], position: int) -> Optional[tuple[PageInfo, int]]:
        """전체 텍스트 위치에서 페이지와 상대 위치 찾기"""
        current_pos = 0
        
        for page in pages:
            page_length = len(page.raw_text) + 1
            
            if current_pos <= position < current_pos + page_length:
                relative_pos = position - current_pos
                return page, relative_pos
            
            current_pos += page_length
        
        return None
    
    async def _check_grammar_llm(self, doc_meta: DocumentMeta, pages: List[PageInfo]) -> List[Issue]:
        """LLM 기반 문법 검사 - 멀티모달 텍스트 포함"""
        if not self.llm:
            return []
        
        issues = []
        
        for page in pages:
            # 1. 페이지 텍스트 검사
            if page.raw_text.strip() and len(page.raw_text) <= 1000:
                korean_text = self._extract_korean_text(page.raw_text)
                if len(korean_text) >= 20:
                    try:
                        page_issues = await self._check_grammar_for_text(
                            doc_meta, page, korean_text, "페이지 텍스트"
                        )
                        issues.extend(page_issues)
                    except Exception as e:
                        logger.warning(f"페이지 문법 검사 실패: {str(e)}")
            
            # 2. OCR 텍스트 검사
            for element in page.elements:
                if (element.element_type == ElementType.IMAGE and 
                    element.image_data and 
                    element.image_data.ocr_text):
                    
                    korean_text = self._extract_korean_text(element.image_data.ocr_text)
                    if len(korean_text) >= 20:
                        try:
                            ocr_issues = await self._check_grammar_for_text(
                                doc_meta, page, korean_text, f"OCR 텍스트 ({element.element_id})"
                            )
                            # OCR 이슈는 element_id 포함
                            for issue in ocr_issues:
                                issue.element_id = element.element_id
                                issue.message = f"[OCR] {issue.message}"
                            issues.extend(ocr_issues)
                        except Exception as e:
                            logger.warning(f"OCR 문법 검사 실패: {str(e)}")
        
        logger.info(f"LLM 문법 검사 (멀티모달): {len(issues)}개 발견")
        return issues
    
    async def _check_grammar_for_text(self, doc_meta: DocumentMeta, page: PageInfo, text: str, source: str) -> List[Issue]:
        """특정 텍스트에 대한 문법 검사"""
        try:
            prompt = self._create_grammar_check_prompt(text)
            response = await self.llm.acomplete(prompt)
            
            page_issues = self._parse_grammar_response(response.text, doc_meta, page)
            
            # API 호출 제한
            await asyncio.sleep(0.5)
            
            return page_issues
            
        except Exception as e:
            logger.warning(f"{source} 문법 검사 실패: {str(e)}")
            return []
    
    def _extract_korean_text(self, text: str) -> str:
        """텍스트에서 한국어 부분만 추출"""
        sentences = re.split(r'[.!?]\s*', text)
        korean_sentences = []
        
        for sentence in sentences:
            if re.search(r'[가-힣]', sentence) and len(sentence.strip()) > 5:
                korean_sentences.append(sentence.strip())
        
        return ' '.join(korean_sentences)
    
    def _create_grammar_check_prompt(self, text: str) -> str:
        """문법 검사용 프롬프트 생성"""
        return f"""다음 한국어 텍스트의 문법과 맞춤법을 검사해주세요.

텍스트: {text}

다음 형식으로 간단히 응답해주세요:
- 문제가 있으면: "오류발견: [문제있는부분] -> [수정제안]"
- 문제가 없으면: "문제없음"

여러 오류가 있다면 각각 한 줄씩 작성해주세요."""
    
    def _parse_grammar_response(self, response: str, doc_meta: DocumentMeta, page: PageInfo) -> List[Issue]:
        """LLM 응답에서 문법 이슈 파싱"""
        issues = []
        
        if "문제없음" in response or "오류발견" not in response:
            return issues
        
        lines = response.strip().split('\n')
        
        for line in lines:
            if "오류발견:" in line:
                try:
                    parts = line.split("오류발견:")[1].strip()
                    if "->" in parts:
                        original, suggestion = parts.split("->", 1)
                        original = original.strip()
                        suggestion = suggestion.strip()
                        
                        pos = page.raw_text.find(original)
                        if pos != -1:
                            location = TextLocation(
                                start=pos,
                                end=pos + len(original)
                            )
                            
                            issue_id = generate_issue_id(
                                doc_meta.doc_id,
                                page.page_id,
                                location,
                                IssueType.GRAMMAR
                            )
                            
                            issue = Issue(
                                issue_id=issue_id,
                                doc_id=doc_meta.doc_id,
                                page_id=page.page_id,
                                issue_type=IssueType.GRAMMAR,
                                text_location=location,
                                original_text=original,
                                message="LLM이 감지한 문법/맞춤법 오류",
                                suggestion=suggestion,
                                confidence=0.75,
                                confidence_level="medium",
                                agent_name="multimodal_quality_agent"
                            )
                            
                            issues.append(issue)
                
                except Exception as e:
                    logger.warning(f"문법 응답 파싱 실패: {str(e)}")
        
        return issues


# 테스트 함수
async def test_multimodal_quality_agent():
    """MultimodalQualityAgent 테스트"""
    print("🧪 MultimodalQualityAgent 테스트 시작...")
    
    import os
    from dotenv import load_dotenv
    from pathlib import Path
    env_path = Path(__file__).resolve().parents[2] / '.env.dev'
    load_dotenv(env_path)
    api_key = os.getenv("OPENAI_API_KEY")
    
    # 에이전트 생성
    agent = MultimodalQualityAgent(
        openai_api_key=api_key,
        enable_vision_analysis=bool(api_key)
    )
    
    # 테스트용 멀티모달 문서 생성
    from src.core.models import (
        DocumentMeta, PageInfo, PageElement, ElementType,
        ImageElement, TableElement, BoundingBox, generate_doc_id
    )
    
    # 문서 메타데이터
    doc_meta = DocumentMeta(
        doc_id=generate_doc_id("test_multimodal.pdf"),
        title="멀티모달 테스트 문서",
        doc_type="pdf",
        total_pages=1,
        file_path="test_multimodal.pdf"
    )
    
    # 테스트 페이지 생성
    test_elements = []
    
    # 1. 텍스트 요소 (오탈자 포함)
    text_element = PageElement(
        element_id="p001_text_001",
        element_type=ElementType.TEXT,
        text_content="딥러닝과 심층학습을 사용한 알고리듬 연구입니다. 데이타 전처리가 중요합니다.",
        confidence=1.0
    )
    test_elements.append(text_element)
    
    # 2. 이미지 요소 (OCR 텍스트에 오탈자 포함)
    image_element = PageElement(
        element_id="p001_image_001",
        element_type=ElementType.IMAGE,
        bbox=BoundingBox(x=100, y=200, width=200, height=150),  # 작은 크기
        image_data=ImageElement(
            element_id="p001_image_001",
            bbox=BoundingBox(x=100, y=200, width=200, height=150),
            format="bmp",  # 권장하지 않는 형식
            size_bytes=8 * 1024 * 1024,  # 8MB (큰 파일)
            dimensions=(200, 150),  # 작은 해상도
            ocr_text="머신러닝과 기계학습은 동일한 개념입니다",  # 용어 불일치
            description="신경망 구조 다이어그램"
        ),
        confidence=0.8
    )
    test_elements.append(image_element)
    
    # 3. 표 요소 (품질 문제 포함)
    table_element = PageElement(
        element_id="p001_table_001",
        element_type=ElementType.TABLE,
        bbox=BoundingBox(x=50, y=400, width=400, height=100),
        table_data=TableElement(
            element_id="p001_table_001",
            headers=[],  # 헤더 없음
            rows=[
                ["", "0.95", "빠름"],  # 빈 셀 포함
                ["Random Forest", "", "중간"],  # 빈 셀 포함
                ["Neural Network", "0.97"]  # 열 개수 불일치
            ]
        ),
        confidence=0.9
    )
    test_elements.append(table_element)
    
    # 4. 차트 요소 (가독성 문제 포함)
    chart_element = PageElement(
        element_id="p001_chart_001",
        element_type=ElementType.CHART,
        bbox=BoundingBox(x=50, y=550, width=400, height=200),
        chart_data=ChartElement(
            element_id="p001_chart_001",
            chart_type="bar",
            title="",  # 제목 없음
            x_label="",  # X축 라벨 없음
            y_label="",  # Y축 라벨 없음
            data_points=[{"x": 1, "y": 0.95}],  # 데이터 포인트 부족
            legend=[]  # 범례 없음
        ),
        confidence=0.7
    )
    test_elements.append(chart_element)
    
    # 페이지 생성
    test_page = PageInfo(
        page_id="p001",
        page_number=1,
        raw_text="딥러닝과 심층학습을 사용한 알고리듬 연구입니다. 데이타 전처리가 중요합니다.",
        word_count=12,
        elements=test_elements
    )
    
    # 멀티모달 품질 검사 실행
    print("\n🔍 멀티모달 품질 검사 실행 중...")
    issues = await agent.check_document(doc_meta, [test_page])
    
    print(f"\n📋 발견된 이슈들 ({len(issues)}개):")
    
    # 이슈 타입별 분류
    issue_by_type = {}
    for issue in issues:
        issue_type = issue.issue_type.value
        if issue_type not in issue_by_type:
            issue_by_type[issue_type] = []
        issue_by_type[issue_type].append(issue)
    
    for issue_type, type_issues in issue_by_type.items():
        print(f"\n📌 {issue_type.upper()} ({len(type_issues)}개)")
        
        for issue in type_issues:
            print(f"   🔍 {issue.original_text[:30]}...")
            print(f"      메시지: {issue.message}")
            print(f"      제안: {issue.suggestion}")
            print(f"      신뢰도: {issue.confidence:.2f}")
            if issue.element_id:
                print(f"      요소 ID: {issue.element_id}")
            print(f"      위치: {issue.page_id}")
    
    print(f"\n📊 이슈 요약:")
    print(f"   텍스트 이슈: {len(issue_by_type.get('typo', []) + issue_by_type.get('consistency', []) + issue_by_type.get('grammar', []))}개")
    print(f"   이미지 품질: {len(issue_by_type.get('image_quality', []))}개")
    print(f"   표 형식: {len(issue_by_type.get('table_format', []))}개")
    print(f"   차트 가독성: {len(issue_by_type.get('chart_readability', []))}개")
    
    print("\n🎉 MultimodalQualityAgent 테스트 완료!")


if __name__ == "__main__":
    asyncio.run(test_multimodal_quality_agent())