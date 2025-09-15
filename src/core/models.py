"""
EDU-Audit Core Models
핵심 데이터 구조 정의 - MCP 도구 인터페이스에 맞춘 설계
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo
import json

class IssueType(str, Enum):
    """발견 가능한 이슈 요형"""
    TYPO = "typo"
    CONSISTENCY = "consistency"
    FACT = "fact"
    GRAMMAR = "grammar"

class ConfidenceLevel(str, Enum):
    """신뢰도 수준"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TextLocation(BaseModel):
    """텍스트 위치 정보"""
    start: int = Field(..., ge=0, description="시작 위치")
    end: int = Field(..., ge=0, description="종료 위치")

    @field_validator('end')
    def end_after_start(cls, v, info: ValidationInfo):
        if (start := info.data.get('start')) is not None and v <= start:
            raise ValueError('end must be greater than start')
        return v

class PageInfo(BaseModel):
    """페이지 정보"""
    page_id: str = Field(..., description="페이지 ID (예: p001, slide_03)")
    page_number: int = Field(..., ge=1, description="페이지 번호")
    raw_text: str = Field(default="", description="원본 텍스트")
    word_count: int = Field(default=0, ge=0, description="단어 수")


class DocumentMeta(BaseModel):
    """문서 메타데이터"""
    doc_id: str = Field(..., description="문서 고유 ID")
    title: Optional[str] = Field(None, description="문서 제목")
    doc_type: str = Field(..., description="문서 타입 (pdf, ppt)")
    total_pages: int = Field(..., ge=1, description="총 페이지 수")
    file_path: str = Field(..., description="파일 경로")
    created_at: datetime = Field(default_factory=datetime.now)


class Issue(BaseModel):
    """발견된 이슈"""
    issue_id: str = Field(..., description="이슈 고유 ID")
    doc_id: str = Field(..., description="문서 ID")
    page_id: str = Field(..., description="페이지 ID")
    
    issue_type: IssueType = Field(..., description="이슈 유형")
    location: TextLocation = Field(..., description="텍스트 위치")
    
    original_text: str = Field(..., description="원본 텍스트")
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
    
    # 모델 전체 검증 후 total_issues 자동 계산
    @model_validator(mode="after")
    def set_total_issues(self) -> "AuditReport":
        self.total_issues = len(self.issues)
        return self


# MCP 도구 입출력 모델
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


# 유틸리티 함수들
def generate_doc_id(file_path: str) -> str:
    """파일 경로에서 문서 ID 생성"""
    import hashlib
    from pathlib import Path
    
    path = Path(file_path)
    content = f"{path.name}_{path.stat().st_mtime if path.exists() else 'unknown'}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def generate_issue_id(doc_id: str, page_id: str, location: TextLocation, issue_type: IssueType) -> str:
    """이슈 ID 생성"""
    import hashlib
    
    content = f"{doc_id}_{page_id}_{location.start}_{issue_type.value}"
    return hashlib.md5(content.encode()).hexdigest()[:8]


def generate_report_id(doc_id: str) -> str:
    """보고서 ID 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"report_{doc_id}_{timestamp}"


# 검증용 더미 데이터 생성 함수 (테스트용)
def create_sample_issue(doc_id: str = "test_doc") -> Issue:
    """테스트용 샘플 이슈 생성"""
    location = TextLocation(start=10, end=20)
    return Issue(
        issue_id=generate_issue_id(doc_id, "p001", location, IssueType.TYPO),
        doc_id=doc_id,
        page_id="p001",
        issue_type=IssueType.TYPO,
        location=location,
        original_text="알고리듬",
        message="표기법 오류",
        suggestion="알고리즘",
        confidence=0.95,
        confidence_level=ConfidenceLevel.HIGH,
        agent_name="quality_agent"
    )


def create_sample_report(doc_id: str = "test_doc") -> AuditReport:
    """테스트용 샘플 보고서 생성"""
    sample_issue = create_sample_issue(doc_id)
    return AuditReport(
        report_id=generate_report_id(doc_id),
        doc_id=doc_id,
        issues=[sample_issue],
        agents_used=["quality_agent"]
    )


# 모델 검증 테스트
if __name__ == "__main__":
    # 기본 모델 테스트
    print("🧪 데이터 모델 테스트 시작...")
    
    # 1. Issue 모델 테스트
    issue = create_sample_issue()
    print(f"✅ Issue 생성: {issue.issue_id}")
    print(f"   신뢰도 레벨: {issue.confidence_level}")
    
    # 2. AuditReport 모델 테스트
    report = create_sample_report()
    print(f"✅ Report 생성: {report.report_id}")
    print(f"   총 이슈 수: {report.total_issues}")
    
    # 3. JSON 직렬화
    report_json = report.model_dump_json(indent=2)
    print("✅ JSON 직렬화 성공")

    # 4. 역직렬화
    restored_report = AuditReport.model_validate_json(report_json)
    print(f"✅ JSON 역직렬화 성공: {restored_report.report_id}")
    
    print("🎉 모든 모델 테스트 통과!")