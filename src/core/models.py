"""
EDU-Audit Core Models - Multimodal Extension
핵심 데이터 구조 정의 - 멀티모달 기능 추가
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo
import json
import base64

class IssueType(str, Enum):
    """발견 가능한 이슈 유형"""
    TYPO = "typo"
    CONSISTENCY = "consistency"
    FACT = "fact"
    GRAMMAR = "grammar"
    # 멀티모달 이슈 타입 추가
    IMAGE_QUALITY = "image_quality"
    TABLE_FORMAT = "table_format"
    CHART_READABILITY = "chart_readability"

class ConfidenceLevel(str, Enum):
    """신뢰도 수준"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ElementType(str, Enum):
    """페이지 요소 타입"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    EQUATION = "equation"
    SHAPE = "shape"

class TextLocation(BaseModel):
    """텍스트 위치 정보"""
    start: int = Field(..., ge=0, description="시작 위치")
    end: int = Field(..., ge=0, description="종료 위치")

    @field_validator('end')
    def end_after_start(cls, v, info: ValidationInfo):
        if (start := info.data.get('start')) is not None and v <= start:
            raise ValueError('end must be greater than start')
        return v

class BoundingBox(BaseModel):
    """이미지/표/차트 등의 위치 정보"""
    x: float = Field(..., ge=0, description="X 좌표")
    y: float = Field(..., ge=0, description="Y 좌표")
    width: float = Field(..., gt=0, description="너비")
    height: float = Field(..., gt=0, description="높이")
    
    def area(self) -> float:
        """영역 크기 계산"""
        return self.width * self.height

class ImageElement(BaseModel):
    """이미지 요소"""
    element_id: str = Field(..., description="요소 ID")
    bbox: BoundingBox = Field(..., description="위치 정보")
    
    # 이미지 메타데이터
    format: str = Field(..., description="이미지 포맷 (jpg, png, etc)")
    size_bytes: int = Field(..., ge=0, description="파일 크기")
    dimensions: Tuple[int, int] = Field(..., description="이미지 크기 (width, height)")
    
    # 추출된 정보
    ocr_text: Optional[str] = Field(None, description="OCR로 추출한 텍스트")
    description: Optional[str] = Field(None, description="AI가 생성한 이미지 설명")
    alt_text: Optional[str] = Field(None, description="원본 alt 텍스트")
    
    # 이미지 데이터 (선택적)
    image_data: Optional[str] = Field(None, description="Base64 인코딩된 이미지 데이터")
    
    @field_validator('image_data')
    def validate_image_data(cls, v):
        if v and not v.startswith('data:image/'):
            # Base64 데이터만 있는 경우 data URL 형태로 변환
            return f"data:image/jpeg;base64,{v}"
        return v

class TableElement(BaseModel):
    """표 요소"""
    element_id: str = Field(..., description="요소 ID")
    bbox: Optional[BoundingBox] = Field(None, description="위치 정보")
    
    # 표 구조
    headers: List[str] = Field(default_factory=list, description="헤더 행")
    rows: List[List[str]] = Field(default_factory=list, description="데이터 행들")
    row_count: int = Field(default=0, ge=0, description="행 수")
    col_count: int = Field(default=0, ge=0, description="열 수")
    
    # 표 품질 메트릭
    has_headers: bool = Field(default=False, description="헤더 존재 여부")
    is_structured: bool = Field(default=True, description="구조화 여부")
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="추출 신뢰도")
    
    @model_validator(mode="after")
    def set_dimensions(self) -> "TableElement":
        self.row_count = len(self.rows)
        self.col_count = len(self.headers) if self.headers else (len(self.rows[0]) if self.rows else 0)
        self.has_headers = bool(self.headers)
        return self

class ChartElement(BaseModel):
    """차트/그래프 요소"""
    element_id: str = Field(..., description="요소 ID")
    bbox: Optional[BoundingBox] = Field(None, description="위치 정보")
    
    # 차트 정보
    chart_type: str = Field(..., description="차트 타입 (bar, line, pie, etc)")
    title: Optional[str] = Field(None, description="차트 제목")
    x_label: Optional[str] = Field(None, description="X축 라벨")
    y_label: Optional[str] = Field(None, description="Y축 라벨")
    
    # 추출된 데이터
    data_points: List[Dict[str, Any]] = Field(default_factory=list, description="데이터 포인트")
    legend: List[str] = Field(default_factory=list, description="범례")
    
    # AI 분석 결과
    description: Optional[str] = Field(None, description="차트 내용 설명")
    insights: List[str] = Field(default_factory=list, description="도출된 인사이트")

class PageElement(BaseModel):
    """페이지 내 개별 요소 (통합)"""
    element_id: str = Field(..., description="요소 ID")
    element_type: ElementType = Field(..., description="요소 타입")
    bbox: Optional[BoundingBox] = Field(None, description="위치 정보")
    
    # 텍스트 요소
    text_content: Optional[str] = Field(None, description="텍스트 내용")
    
    # 멀티모달 요소들 (Union 타입 대신 개별 필드)
    image_data: Optional[ImageElement] = Field(None, description="이미지 데이터")
    table_data: Optional[TableElement] = Field(None, description="표 데이터")
    chart_data: Optional[ChartElement] = Field(None, description="차트 데이터")
    
    # 메타데이터
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="추출 신뢰도")
    processing_notes: List[str] = Field(default_factory=list, description="처리 과정 노트")

class PageInfo(BaseModel):
    """확장된 페이지 정보 - 멀티모달 지원"""
    page_id: str = Field(..., description="페이지 ID (예: p001, slide_03)")
    page_number: int = Field(..., ge=1, description="페이지 번호")
    
    # 기존 텍스트 정보
    raw_text: str = Field(default="", description="원본 텍스트")
    word_count: int = Field(default=0, ge=0, description="단어 수")
    
    # 멀티모달 요소들
    elements: List[PageElement] = Field(default_factory=list, description="페이지 내 모든 요소")
    
    # 페이지 레벨 통계
    image_count: int = Field(default=0, ge=0, description="이미지 수")
    table_count: int = Field(default=0, ge=0, description="표 수")
    chart_count: int = Field(default=0, ge=0, description="차트 수")
    
    # 페이지 레이아웃 정보
    layout_type: Optional[str] = Field(None, description="레이아웃 타입 (title, content, mixed)")
    
    @model_validator(mode="after")
    def update_counts(self) -> "PageInfo":
        """요소 수 자동 계산"""
        self.image_count = sum(1 for e in self.elements if e.element_type == ElementType.IMAGE)
        self.table_count = sum(1 for e in self.elements if e.element_type == ElementType.TABLE)
        self.chart_count = sum(1 for e in self.elements if e.element_type == ElementType.CHART)
        return self

# 기존 모델들은 그대로 유지...
class DocumentMeta(BaseModel):
    """문서 메타데이터"""
    doc_id: str = Field(..., description="문서 고유 ID")
    title: Optional[str] = Field(None, description="문서 제목")
    doc_type: str = Field(..., description="문서 타입 (pdf, ppt)")
    total_pages: int = Field(..., ge=1, description="총 페이지 수")
    file_path: str = Field(..., description="파일 경로")
    created_at: datetime = Field(default_factory=datetime.now)
    
    # 멀티모달 통계 추가
    total_images: int = Field(default=0, ge=0, description="전체 이미지 수")
    total_tables: int = Field(default=0, ge=0, description="전체 표 수")
    total_charts: int = Field(default=0, ge=0, description="전체 차트 수")

class Issue(BaseModel):
    """발견된 이슈 - 멀티모달 지원"""
    issue_id: str = Field(..., description="이슈 고유 ID")
    doc_id: str = Field(..., description="문서 ID")
    page_id: str = Field(..., description="페이지 ID")
    
    issue_type: IssueType = Field(..., description="이슈 유형")
    
    # 위치 정보 - 텍스트 또는 바운딩박스
    text_location: Optional[TextLocation] = Field(None, description="텍스트 위치")
    bbox_location: Optional[BoundingBox] = Field(None, description="요소 위치")
    element_id: Optional[str] = Field(None, description="관련 요소 ID")
    
    original_text: str = Field(..., description="원본 텍스트/설명")
    message: str = Field(..., description="이슈 설명")
    suggestion: Optional[str] = Field(None, description="수정 제안")
    
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도")
    confidence_level: ConfidenceLevel = Field(..., description="신뢰도 레벨")
    
    detected_at: datetime = Field(default_factory=datetime.now)
    agent_name: str = Field(..., description="검출한 에이전트명")
    
    @field_validator('confidence_level', mode='before')
    def set_confidence_level(cls, v, info: ValidationInfo):
        confidence = info.data.get('confidence')
        if confidence is None:
            return v
        if confidence >= 0.9:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

class AuditReport(BaseModel):
    """검수 보고서"""
    report_id: str = Field(..., description="보고서 ID")
    doc_id: str = Field(..., description="대상 문서 ID")
    
    issues: List[Issue] = Field(default_factory=list, description="발견된 이슈들")
    total_issues: int = Field(default=0, ge=0, description="총 이슈 수")
    
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = Field(None)
    processing_time: Optional[float] = Field(None, description="처리 시간(초)")
    
    agents_used: List[str] = Field(default_factory=list, description="사용된 에이전트")
    
    # 멀티모달 통계
    text_issues: int = Field(default=0, ge=0, description="텍스트 이슈 수")
    image_issues: int = Field(default=0, ge=0, description="이미지 이슈 수")
    table_issues: int = Field(default=0, ge=0, description="표 이슈 수")
    chart_issues: int = Field(default=0, ge=0, description="차트 이슈 수")
    
    @model_validator(mode="after")
    def calculate_stats(self) -> "AuditReport":
        self.total_issues = len(self.issues)
        
        # 이슈 타입별 통계
        self.text_issues = sum(1 for issue in self.issues 
                              if issue.issue_type in [IssueType.TYPO, IssueType.CONSISTENCY, 
                                                    IssueType.FACT, IssueType.GRAMMAR])
        self.image_issues = sum(1 for issue in self.issues 
                               if issue.issue_type == IssueType.IMAGE_QUALITY)
        self.table_issues = sum(1 for issue in self.issues 
                               if issue.issue_type == IssueType.TABLE_FORMAT)
        self.chart_issues = sum(1 for issue in self.issues 
                               if issue.issue_type == IssueType.CHART_READABILITY)
        
        return self

# MCP 도구 입출력 모델 (기존 유지)
class FactCheckRequest(BaseModel):
    """사실 확인 요청"""
    sentence: str = Field(..., min_length=1, description="확인할 문장")
    context: Optional[str] = Field(None, description="문맥 정보")

class FactCheckResult(BaseModel):
    """사실 확인 결과"""
    sentence: str = Field(..., description="확인한 문장")
    is_factual: bool = Field(..., description="사실 여부")
    confidence: float = Field(..., ge=0.0, le=1.0, description="확신도")
    explanation: str = Field(..., description="판단 근거")
    sources: List[str] = Field(default_factory=list, description="참조 소스")
    checked_at: datetime = Field(default_factory=datetime.now)

class QueryRequest(BaseModel):
    """자연어 질의 요청"""
    question: str = Field(..., min_length=1, description="사용자 질문")
    doc_id: Optional[str] = Field(None, description="대상 문서 ID")

class QueryResponse(BaseModel):
    """자연어 질의 응답"""
    question: str = Field(..., description="원본 질문")
    answer: str = Field(..., description="답변")
    relevant_issues: List[Issue] = Field(default_factory=list, description="관련 이슈들")
    confidence: float = Field(..., ge=0.0, le=1.0, description="답변 신뢰도")
    generated_at: datetime = Field(default_factory=datetime.now)

# 유틸리티 함수들 (기존 + 새로 추가)
def generate_doc_id(file_path: str) -> str:
    """파일 경로에서 문서 ID 생성"""
    import hashlib
    from pathlib import Path
    
    path = Path(file_path)
    content = f"{path.name}_{path.stat().st_mtime if path.exists() else 'unknown'}"
    return hashlib.md5(content.encode()).hexdigest()[:12]

def generate_element_id(page_id: str, element_type: ElementType, index: int) -> str:
    """페이지 요소 ID 생성"""
    return f"{page_id}_{element_type.value}_{index:03d}"

def generate_issue_id(doc_id: str, page_id: str, location: Union[TextLocation, BoundingBox], issue_type: IssueType) -> str:
    """이슈 ID 생성"""
    import hashlib
    
    if isinstance(location, TextLocation):
        loc_str = f"{location.start}_{location.end}"
    else:  # BoundingBox
        loc_str = f"{location.x}_{location.y}"
    
    content = f"{doc_id}_{page_id}_{loc_str}_{issue_type.value}"
    return hashlib.md5(content.encode()).hexdigest()[:8]

def generate_report_id(doc_id: str) -> str:
    """보고서 ID 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"report_{doc_id}_{timestamp}"

# 멀티모달 샘플 데이터 생성 함수
def create_sample_multimodal_page() -> PageInfo:
    """멀티모달 샘플 페이지 생성"""
    # 텍스트 요소
    text_element = PageElement(
        element_id="p001_text_001",
        element_type=ElementType.TEXT,
        text_content="머신러닝 개요",
        confidence=1.0
    )
    
    # 이미지 요소
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
            description="신경망 구조 다이어그램",
            ocr_text="Input Layer -> Hidden Layer -> Output Layer"
        )
    )
    
    # 표 요소  
    table_element = PageElement(
        element_id="p001_table_001",
        element_type=ElementType.TABLE,
        bbox=BoundingBox(x=50, y=450, width=400, height=150),
        table_data=TableElement(
            element_id="p001_table_001",
            headers=["알고리즘", "정확도", "속도"],
            rows=[
                ["SVM", "0.95", "빠름"],
                ["Random Forest", "0.93", "중간"],
                ["Neural Network", "0.97", "느림"]
            ]
        )
    )
    
    return PageInfo(
        page_id="p001",
        page_number=1,
        raw_text="머신러닝 개요\n\n다양한 알고리즘 비교...",
        word_count=20,
        elements=[text_element, image_element, table_element]
    )

# 검증용 테스트
if __name__ == "__main__":
    print("🧪 멀티모달 데이터 모델 테스트 시작...")
    
    # 멀티모달 페이지 생성 테스트
    page = create_sample_multimodal_page()
    print(f"✅ 멀티모달 페이지 생성: {page.page_id}")
    print(f"   요소 수: {len(page.elements)}")
    print(f"   이미지: {page.image_count}, 표: {page.table_count}, 차트: {page.chart_count}")
    
    # JSON 직렬화 테스트
    page_json = page.model_dump_json(indent=2)
    print("✅ JSON 직렬화 성공")
    
    # 역직렬화 테스트
    restored_page = PageInfo.model_validate_json(page_json)
    print(f"✅ JSON 역직렬화 성공: {restored_page.page_id}")
    
    print("🎉 멀티모달 모델 테스트 통과!")