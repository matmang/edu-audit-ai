# app/agents/quality_wrapper.py
"""
QualityAgent Wrapper - 품질 검수 에이전트 래퍼
"""

import logging
from typing import Any, Dict, List, Optional

from src.app.agents.base import BaseAgentWrapper, validate_doc_id
from src.agents.quality_agent import QualityAgent
from src.agents.document_agent import DocumentAgent
from src.core.models import generate_report_id

logger = logging.getLogger(__name__)


class QualityAgentWrapper(BaseAgentWrapper):
    """
    QualityAgent를 위한 래퍼 클래스
    
    지원하는 액션:
    - analyze: 문서 품질 분석 (오탈자, 문법, 가독성 등)
    - summary: 품질 분석 요약 정보
    - config: 현재 설정 조회
    - clear_cache: 특정 문서의 캐시 삭제
    """
    
    def __init__(self, quality_agent: QualityAgent, document_agent: DocumentAgent):
        super().__init__("quality", quality_agent)
        self.quality_agent = quality_agent
        self.document_agent = document_agent
        self._analysis_cache = {}
        logger.info("QualityAgentWrapper 초기화 완료")

    def get_supported_actions(self) -> List[str]:
        """지원하는 액션 목록"""
        return ["analyze", "summary", "config", "clear_cache"]

    async def _execute_action(self, action: str, request: Dict[str, Any]) -> Any:
        """실제 액션 실행 로직"""
        
        if action == "analyze":
            return await self._analyze_quality(request)
        elif action == "summary":
            return await self._get_summary(request)
        elif action == "config":
            return await self._get_config(request)
        elif action == "clear_cache":
            return await self._clear_cache(request)
        else:
            raise ValueError(f"지원하지 않는 액션: {action}")

    async def _analyze_quality(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        품질 분석 액션
        
        Args:
            request: {"doc_id": "document_id"}
            
        Returns:
            Dict: 품질 분석 결과 (이슈 목록)
        """
        doc_id = validate_doc_id(request.get("doc_id"))
        force_reanalyze = request.get("force_reanalyze", False)
        
        # 문서 존재 확인
        if not self.document_agent.get_document(doc_id):
            raise ValueError(f"문서를 찾을 수 없습니다: {doc_id}")
        
        if not force_reanalyze and doc_id in self._analysis_cache:
            cached_result = self._analysis_cache[doc_id]
            logger.info(f"캐시된 품질 분석 결과 사용: {doc_id}")

            cached_issues = cached_result["issue"]
            issues_data = []
            for issue in cached_issues:
                if hasattr(issue, 'model_dump'):
                    issue_dict = issue.model_dump(made='json')
                else:
                    issue_dict = issue
                issues_data.append(issue_dict)
            
            return {
                "report_id": cached_result["report_id"],
                "doc_id": doc_id,
                "issues": issues_data,
                "total_issues": len(cached_issues),
                "analysis_type": "quality",
                "from_cache": True,
                "cached_at": cached_result["timestamp"].isoformat(),
                "message": f"캐시된 품질 분석 결과: {len(cached_issues)}개 이슈"
            }

        logger.info(f"품질 분석 시작: {doc_id}")
        
        try:
            # QualityAgent로 실제 분석 수행
            issues = await self.quality_agent.analyze_document(self.document_agent, doc_id)
            report_id = generate_report_id(doc_id)

            logger.info(f"이슈 정상적으로 생성 \n{issues[0]}")

            from datetime import datetime
            self._analysis_cache[doc_id] = {
                "issues": issues,
                "timestamp": datetime.now(),
                "report_id": report_id
            }
            
            logger.info(f"품질 분석 완료 및 캐시 저장: {len(issues)}개 이슈 발견")
            
            # Issue 객체들을 JSON 직렬화 가능한 형태로 변환
            issues_data = []
            for issue in issues:
                issue_dict = issue.model_dump(mode='json')
                issues_data.append(issue_dict)
            
            logger.info(f"json 직렬화 성공")
            
            return {
                "report_id": report_id,
                "doc_id": doc_id,
                "issues": issues_data,
                "total_issues": len(issues),
                "analysis_type": "quality",
                "message": f"품질 분석 완료: {len(issues)}개 이슈 발견"
            }
            
        except Exception as e:
            print(e)
            logger.error(f"품질 분석 실패: {str(e)}")
            raise RuntimeError(f"품질 분석 중 오류 발생: {str(e)}")

    async def _get_summary(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        품질 분석 요약 액션
        
        Args:
            request: {"doc_id": "document_id"}
            
        Returns:
            Dict: 품질 분석 요약
        """
        doc_id = validate_doc_id(request.get("doc_id"))
        
        # 문서 존재 확인
        if not self.document_agent.get_document(doc_id):
            raise ValueError(f"문서를 찾을 수 없습니다: {doc_id}")
        
        try:
            # 캐시된 분석 결과가 있는지 확인
            if doc_id in self._analysis_cache:
                cached_result = self._analysis_cache[doc_id]
                cached_issues = cached_result["issues"]

                logger.info(f"캐시된 분석 결과로 요약 생성: {doc_id}")

                summary = self.quality_agent.get_quality_summary(cached_issues)

                return {
                    "doc_id": doc_id,
                    "summary": summary,
                    "analysis_type": "quality_summary",
                    "from_cache": True,
                    "cached_at": cached_result["timestamp"].isoformat(),
                    "message": f"품질 요약 (캐시 사용): 총 {summary['total_issues']}개 이슈"
                }
            else:
                # 품질 분석 실행
                issues = await self.quality_agent.analyze_document(self.document_agent, doc_id)

                # 요약 정보 생성
                summary = self.quality_agent.get_quality_summary(issues)

                logger.info(f"품질 요약 생성 완료: 점수 {summary['quality_score']:.2f}")

                return {
                    "doc_id": doc_id,
                    "summary": summary,
                    "analysis_type": "quality_summary",
                    "message": f"품질 요약 생성 완료: 총 {summary['total_issues']}개 이슈"
                }
            
        except Exception as e:
            logger.error(f"품질 요약 생성 실패: {str(e)}")
            raise RuntimeError(f"품질 요약 생성 중 오류 발생: {str(e)}")

    async def _get_config(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        QualityAgent 설정 조회 액션
        
        Args:
            request: {} (파라미터 없음)
            
        Returns:
            Dict: 현재 QualityAgent 설정
        """
        try:
            config = self.quality_agent.config
            
            config_dict = {
                "max_issues_per_slide": config.max_issues_per_slide,
                "confidence_threshold": config.confidence_threshold,
                "enable_vision_analysis": config.enable_vision_analysis,
                "issue_severity_filter": config.issue_severity_filter,
                "exclude_minor_issues": config.exclude_minor_issues
            }
            
            return {
                "config": config_dict,
                "message": "QualityAgent 설정 조회 완료"
            }
            
        except Exception as e:
            logger.error(f"설정 조회 실패: {str(e)}")
            raise RuntimeError(f"설정 조회 중 오류 발생: {str(e)}")

    async def _clear_cache(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        캐시 삭제 액션
        
        Args:
            request: {
                "doc_id": "document_id"  # 선택사항, 없으면 전체 캐시 삭제
            }
            
        Returns:
            Dict: 캐시 삭제 결과
        """
        doc_id = request.get("doc_id")
        
        try:
            if doc_id:
                # 특정 문서의 캐시만 삭제
                if doc_id in self._analysis_cache:
                    del self._analysis_cache[doc_id]
                    logger.info(f"문서 {doc_id}의 캐시 삭제 완료")
                    return {
                        "doc_id": doc_id,
                        "message": f"문서 {doc_id}의 품질 분석 캐시가 삭제되었습니다."
                    }
                else:
                    return {
                        "doc_id": doc_id,
                        "message": f"문서 {doc_id}의 캐시가 존재하지 않습니다."
                    }
            else:
                # 전체 캐시 삭제
                cache_count = len(self._analysis_cache)
                self._analysis_cache.clear()
                logger.info(f"전체 품질 분석 캐시 삭제 완료: {cache_count}개")
                return {
                    "message": f"전체 품질 분석 캐시가 삭제되었습니다. ({cache_count}개 문서)"
                }
                
        except Exception as e:
            logger.error(f"캐시 삭제 실패: {str(e)}")
            raise RuntimeError(f"캐시 삭제 중 오류 발생: {str(e)}")

    def get_cache_info(self) -> Dict[str, Any]:
        """캐시 정보 조회 (디버깅용)"""
        cache_info = {}
        for doc_id, cached_data in self._analysis_cache.items():
            cache_info[doc_id] = {
                "timestamp": cached_data["timestamp"].isoformat(),
                "issues_count": len(cached_data["issues"]),
                "report_id": cached_data["report_id"]
            }
        
        return {
            "cache_count": len(self._analysis_cache),
            "cached_documents": cache_info
        }

    def _validate_request(self, request: Dict[str, Any]) -> None:
        """QualityAgent 전용 요청 유효성 검사"""
        super()._validate_request(request)
        
        action = request["action"]
        
        # 액션별 추가 유효성 검사
        if action in ["analyze", "summary"]:
            if "doc_id" not in request:
                raise ValueError(f"{action} 액션은 doc_id가 필요합니다")

    def get_agent_specific_info(self) -> Dict[str, Any]:
        """QualityAgent 관련 추가 정보"""
        try:
            config = self.quality_agent.config
            cache_info = self.get_cache_info()
            
            return {
                "vision_model": getattr(self.quality_agent, 'vision_model', 'unknown'),
                "model": getattr(self.quality_agent, 'model', 'unknown'),
                "confidence_threshold": config.confidence_threshold,
                "max_issues_per_slide": config.max_issues_per_slide,
                "issue_severity_filter": config.issue_severity_filter,
                "vision_analysis_enabled": config.enable_vision_analysis,
                "excluded_minor_issues": len(config.exclude_minor_issues),
                "supported_issue_types": [
                    "typo", "grammar", "fact", "image_quality", 
                    "content_clarity", "consistency"
                ],
                "cache_info": cache_info
            }
        except Exception as e:
            logger.warning(f"QualityAgent 추가 정보 조회 실패: {str(e)}")
            return {"error": "추가 정보를 가져올 수 없습니다"}

    def get_info(self) -> Dict[str, Any]:
        """확장된 에이전트 정보"""
        base_info = super().get_info()
        base_info["agent_specific"] = self.get_agent_specific_info()
        return base_info


# 유틸리티 함수들

def validate_quality_config(config_dict: Dict[str, Any]) -> None:
    """품질 검수 설정 유효성 검사"""
    if "confidence_threshold" in config_dict:
        threshold = config_dict["confidence_threshold"]
        if not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
            raise ValueError("confidence_threshold는 0.0-1.0 사이의 숫자여야 합니다")
    
    if "max_issues_per_slide" in config_dict:
        max_issues = config_dict["max_issues_per_slide"]
        if not isinstance(max_issues, int) or max_issues < 1:
            raise ValueError("max_issues_per_slide는 1 이상의 정수여야 합니다")


def get_supported_issue_types() -> List[str]:
    """지원하는 이슈 타입 목록"""
    return ["typo", "grammar", "fact", "image_quality", "content_clarity", "consistency"]


# 테스트용 함수
async def test_quality_wrapper():
    """QualityAgentWrapper 테스트"""
    print("QualityAgentWrapper 테스트 시작...")
    
    # Mock QualityAgent와 DocumentAgent 생성
    class MockQualityAgent:
        def __init__(self):
            from src.agents.quality_agent import QualityConfig
            self.config = QualityConfig()
            self.model = "gpt-5-nano"
            self.vision_model = "gpt-5-nano"
        
        async def analyze_document(self, document_agent, doc_id):
            # Mock 이슈 생성
            from src.core.models import Issue, IssueType, TextLocation, generate_issue_id
            from datetime import datetime
            
            issue = Issue(
                issue_id=generate_issue_id(doc_id, "p001", TextLocation(start=0, end=5), IssueType.TYPO),
                doc_id=doc_id,
                page_id="p001",
                issue_type=IssueType.TYPO,
                text_location=TextLocation(start=0, end=5),
                bbox_location=None,
                element_id=None,
                original_text="테스트 텍스트",
                message="Mock 오탈자 이슈",
                suggestion="수정 제안",
                confidence=0.85,
                confidence_level="high",
                agent_name="quality_agent_test",
                detected_at=datetime.now()
            )
            return [issue]
        
        def get_quality_summary(self, issues):
            return {
                "total_issues": len(issues),
                "quality_score": 0.9,
                "by_type": {"typo": len(issues)},
                "by_severity": {"high": len(issues)},
                "recommendations": ["Mock 권장사항"]
            }
    
    class MockDocumentAgent:
        def get_document(self, doc_id):
            from src.core.models import DocumentMeta
            return DocumentMeta(
                doc_id=doc_id,
                title="테스트 문서",
                doc_type="pdf", 
                total_pages=5,
                file_path="test.pdf"
            ) if doc_id == "test_doc" else None
    
    # Mock 에이전트로 래퍼 테스트
    mock_quality = MockQualityAgent()
    mock_document = MockDocumentAgent()
    wrapper = QualityAgentWrapper(mock_quality, mock_document)
    
    print(f"지원 액션: {wrapper.get_supported_actions()}")
    
    # analyze 액션 테스트
    request = {"action": "analyze", "doc_id": "test_doc"}
    result = await wrapper.handle(request)
    print(f"Analyze 테스트 - 성공: {result['success']}")
    if result['success']:
        print(f"   발견된 이슈: {result['result']['total_issues']}개")
    
    # summary 액션 테스트
    request = {"action": "summary", "doc_id": "test_doc"}
    result = await wrapper.handle(request)
    print(f"Summary 테스트 - 성공: {result['success']}")
    if result['success']:
        print(f"   품질 점수: {result['result']['summary']['quality_score']}")
    
    # config 액션 테스트
    request = {"action": "config"}
    result = await wrapper.handle(request)
    print(f"Config 테스트 - 성공: {result['success']}")
    
    # 에이전트 정보 테스트
    info = wrapper.get_info()
    print(f"에이전트 정보: {info['name']}, 상태: {info['status']}")
    
    print("QualityAgentWrapper 테스트 완료!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_quality_wrapper())