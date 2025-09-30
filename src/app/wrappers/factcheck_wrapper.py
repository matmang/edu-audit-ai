# app/wrappers/factcheck_wrapper.py
"""
FactCheckAgent Wrapper - 팩트체킹 에이전트 래퍼
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.app.wrappers.base import BaseAgentWrapper, validate_doc_id
from src.agents.factcheck_agent import FactCheckAgent
from src.agents.document_agent import DocumentAgent
from src.core.models import generate_report_id

logger = logging.getLogger(__name__)


class FactCheckAgentWrapper(BaseAgentWrapper):
    """
    FactCheckAgent를 위한 래퍼 클래스
    
    지원하는 액션:
    - analyze: 문서 팩트체킹 분석 (선택적 검색 기반)
    - summary: 팩트체킹 분석 요약 정보 (캐시된 결과 우선 사용)
    - config: 현재 설정 조회
    - clear_cache: 특정 문서의 캐시 삭제
    - clear_search_cache: 검색 캐시 삭제
    """
    
    def __init__(self, factcheck_agent: FactCheckAgent, document_agent: DocumentAgent):
        super().__init__("factcheck", factcheck_agent)
        self.factcheck_agent = factcheck_agent
        self.document_agent = document_agent
        
        # 분석 결과 캐시 {doc_id: {"issues": List[Issue], "timestamp": datetime, "report_id": str}}
        self._analysis_cache = {}
        
        logger.info("FactCheckAgentWrapper 초기화 완료")

    def get_supported_actions(self) -> List[str]:
        """지원하는 액션 목록"""
        return ["analyze", "summary", "config", "clear_cache", "clear_search_cache"]

    async def _execute_action(self, action: str, request: Dict[str, Any]) -> Any:
        """실제 액션 실행 로직"""
        
        if action == "analyze":
            return await self._analyze_factcheck(request)
        elif action == "summary":
            return await self._get_summary(request)
        elif action == "config":
            return await self._get_config(request)
        elif action == "clear_cache":
            return await self._clear_cache(request)
        elif action == "clear_search_cache":
            return await self._clear_search_cache(request)
        else:
            raise ValueError(f"지원하지 않는 액션: {action}")

    async def _analyze_factcheck(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        팩트체킹 분석 액션 (결과 캐싱)
        
        Args:
            request: {
                "doc_id": "document_id",
                "force_reanalyze": false  # 선택사항, true면 캐시 무시하고 재분석
            }
            
        Returns:
            Dict: 팩트체킹 분석 결과 (이슈 목록)
        """
        doc_id = validate_doc_id(request.get("doc_id"))
        force_reanalyze = request.get("force_reanalyze", False)
        
        # 문서 존재 확인
        if not self.document_agent.get_document(doc_id):
            raise ValueError(f"문서를 찾을 수 없습니다: {doc_id}")
        
        # 캐시 확인 (강제 재분석이 아닌 경우)
        if not force_reanalyze and doc_id in self._analysis_cache:
            cached_result = self._analysis_cache[doc_id]
            logger.info(f"캐시된 팩트체킹 분석 결과 사용: {doc_id}")
            
            # 캐시된 Issue 객체들을 JSON 직렬화 가능한 형태로 변환
            cached_issues = cached_result["issues"]
            issues_data = []
            for issue in cached_issues:
                if hasattr(issue, 'model_dump'):
                    issue_dict = issue.model_dump(mode='json')
                else:
                    issue_dict = issue  # 이미 dict인 경우
                issues_data.append(issue_dict)
            
            return {
                "report_id": cached_result["report_id"],
                "doc_id": doc_id,
                "issues": issues_data,
                "total_issues": len(cached_issues),
                "analysis_type": "factcheck",
                "from_cache": True,
                "cached_at": cached_result["timestamp"].isoformat(),
                "message": f"캐시된 팩트체킹 분석 결과: {len(cached_issues)}개 이슈"
            }
        
        logger.info(f"팩트체킹 분석 시작: {doc_id}")
        
        try:
            # FactCheckAgent로 실제 분석 수행 (비동기 컨텍스트 매니저 사용)
            async with self.factcheck_agent:
                issues = await self.factcheck_agent.analyze_document(self.document_agent, doc_id)
            
            report_id = generate_report_id(doc_id)
            
            # 분석 결과 캐시에 저장
            self._analysis_cache[doc_id] = {
                "issues": issues,
                "timestamp": datetime.now(),
                "report_id": report_id
            }
            
            logger.info(f"팩트체킹 분석 완료 및 캐시 저장: {len(issues)}개 이슈 발견")
            
            # Issue 객체들을 JSON 직렬화 가능한 형태로 변환
            issues_data = []
            for issue in issues:
                issue_dict = issue.model_dump(mode='json')
                issues_data.append(issue_dict)
            
            return {
                "report_id": report_id,
                "doc_id": doc_id,
                "issues": issues_data,
                "total_issues": len(issues),
                "analysis_type": "factcheck",
                "from_cache": False,
                "message": f"팩트체킹 분석 완료: {len(issues)}개 이슈 발견"
            }
            
        except Exception as e:
            logger.error(f"팩트체킹 분석 실패: {str(e)}")
            raise RuntimeError(f"팩트체킹 분석 중 오류 발생: {str(e)}")

    async def _get_summary(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        팩트체킹 분석 요약 액션 (캐시 우선 사용)
        
        Args:
            request: {"doc_id": "document_id"}
            
        Returns:
            Dict: 팩트체킹 분석 요약
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
                
                logger.info(f"캐시된 분석 결과로 팩트체킹 요약 생성: {doc_id}")
                
                # 요약 정보 생성 (캐시된 이슈들 사용)
                summary = self.factcheck_agent.get_factcheck_summary(cached_issues)
                
                return {
                    "doc_id": doc_id,
                    "summary": summary,
                    "analysis_type": "factcheck_summary",
                    "from_cache": True,
                    "cached_at": cached_result["timestamp"].isoformat(),
                    "message": f"팩트체킹 요약 (캐시 사용): 총 {summary['total_fact_issues']}개 이슈"
                }
            else:
                # 캐시된 분석이 없으면 새로 분석 수행
                logger.info(f"캐시된 분석이 없어 새로 팩트체킹 분석 수행: {doc_id}")
                
                async with self.factcheck_agent:
                    issues = await self.factcheck_agent.analyze_document(self.document_agent, doc_id)
                
                # 분석 결과 캐시에 저장
                self._analysis_cache[doc_id] = {
                    "issues": issues,
                    "timestamp": datetime.now(),
                    "report_id": generate_report_id(doc_id)
                }
                
                # 요약 정보 생성
                summary = self.factcheck_agent.get_factcheck_summary(issues)
                
                logger.info(f"새 팩트체킹 분석 수행 및 요약 생성 완료: 평균 신뢰도 {summary['avg_confidence']:.2f}")
                
                return {
                    "doc_id": doc_id,
                    "summary": summary,
                    "analysis_type": "factcheck_summary",
                    "from_cache": False,
                    "message": f"팩트체킹 요약 (새 분석): 총 {summary['total_fact_issues']}개 이슈"
                }
                
        except Exception as e:
            logger.error(f"팩트체킹 요약 생성 실패: {str(e)}")
            raise RuntimeError(f"팩트체킹 요약 생성 중 오류 발생: {str(e)}")

    async def _get_config(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        FactCheckAgent 설정 조회 액션
        
        Args:
            request: {} (파라미터 없음)
            
        Returns:
            Dict: 현재 FactCheckAgent 설정
        """
        try:
            config_dict = {
                "model": getattr(self.factcheck_agent, 'model', 'unknown'),
                "max_search_results": getattr(self.factcheck_agent, 'max_search_results', 5),
                "search_timeout": getattr(self.factcheck_agent, 'search_timeout', 10),
                "serpapi_enabled": bool(getattr(self.factcheck_agent, 'serpapi_key', None)),
                "search_cache_size": len(getattr(self.factcheck_agent, 'search_cache', {})),
                "verification_cache_size": len(getattr(self.factcheck_agent, 'verification_cache', {})),
                "supported_search_engines": ["SerpAPI", "Mock"] if not getattr(self.factcheck_agent, 'serpapi_key', None) else ["SerpAPI"]
            }
            
            return {
                "config": config_dict,
                "message": "FactCheckAgent 설정 조회 완료"
            }
            
        except Exception as e:
            logger.error(f"설정 조회 실패: {str(e)}")
            raise RuntimeError(f"설정 조회 중 오류 발생: {str(e)}")

    async def _clear_cache(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        분석 결과 캐시 삭제 액션
        
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
                    logger.info(f"문서 {doc_id}의 팩트체킹 캐시 삭제 완료")
                    return {
                        "doc_id": doc_id,
                        "message": f"문서 {doc_id}의 팩트체킹 분석 캐시가 삭제되었습니다."
                    }
                else:
                    return {
                        "doc_id": doc_id,
                        "message": f"문서 {doc_id}의 팩트체킹 캐시가 존재하지 않습니다."
                    }
            else:
                # 전체 캐시 삭제
                cache_count = len(self._analysis_cache)
                self._analysis_cache.clear()
                logger.info(f"전체 팩트체킹 분석 캐시 삭제 완료: {cache_count}개")
                return {
                    "message": f"전체 팩트체킹 분석 캐시가 삭제되었습니다. ({cache_count}개 문서)"
                }
                
        except Exception as e:
            logger.error(f"캐시 삭제 실패: {str(e)}")
            raise RuntimeError(f"캐시 삭제 중 오류 발생: {str(e)}")

    async def _clear_search_cache(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        검색 캐시 삭제 액션 (FactCheckAgent 내부 캐시)
        
        Args:
            request: {} (파라미터 없음)
            
        Returns:
            Dict: 검색 캐시 삭제 결과
        """
        try:
            search_cache = getattr(self.factcheck_agent, 'search_cache', {})
            verification_cache = getattr(self.factcheck_agent, 'verification_cache', {})
            
            search_count = len(search_cache)
            verification_count = len(verification_cache)
            
            # FactCheckAgent의 캐시 삭제
            search_cache.clear()
            verification_cache.clear()
            
            logger.info(f"FactCheckAgent 검색 캐시 삭제 완료: 검색 {search_count}개, 검증 {verification_count}개")
            
            return {
                "message": f"검색 캐시가 삭제되었습니다. (검색: {search_count}개, 검증: {verification_count}개)",
                "search_cache_cleared": search_count,
                "verification_cache_cleared": verification_count
            }
            
        except Exception as e:
            logger.error(f"검색 캐시 삭제 실패: {str(e)}")
            raise RuntimeError(f"검색 캐시 삭제 중 오류 발생: {str(e)}")

    def get_cache_info(self) -> Dict[str, Any]:
        """캐시 정보 조회 (디버깅용)"""
        cache_info = {}
        for doc_id, cached_data in self._analysis_cache.items():
            cache_info[doc_id] = {
                "timestamp": cached_data["timestamp"].isoformat(),
                "issues_count": len(cached_data["issues"]),
                "report_id": cached_data["report_id"]
            }
        
        # FactCheckAgent의 내부 캐시 정보도 포함
        search_cache = getattr(self.factcheck_agent, 'search_cache', {})
        verification_cache = getattr(self.factcheck_agent, 'verification_cache', {})
        
        return {
            "analysis_cache_count": len(self._analysis_cache),
            "cached_documents": cache_info,
            "search_cache_count": len(search_cache),
            "verification_cache_count": len(verification_cache)
        }

    def _validate_request(self, request: Dict[str, Any]) -> None:
        """FactCheckAgent 전용 요청 유효성 검사"""
        super()._validate_request(request)
        
        action = request["action"]
        
        # 액션별 추가 유효성 검사
        if action in ["analyze", "summary"]:
            if "doc_id" not in request:
                raise ValueError(f"{action} 액션은 doc_id가 필요합니다")

    def get_agent_specific_info(self) -> Dict[str, Any]:
        """FactCheckAgent 관련 추가 정보 (캐시 정보 포함)"""
        try:
            cache_info = self.get_cache_info()
            
            return {
                "model": getattr(self.factcheck_agent, 'model', 'unknown'),
                "max_search_results": getattr(self.factcheck_agent, 'max_search_results', 5),
                "search_timeout": getattr(self.factcheck_agent, 'search_timeout', 10),
                "serpapi_enabled": bool(getattr(self.factcheck_agent, 'serpapi_key', None)),
                "supported_analysis_pipeline": [
                    "LLM 필터링", "외부 검색", "결과 대조", "이슈 생성"
                ],
                "supported_issue_categories": [
                    "정확성 문제", "최신성 문제", "검증 불가"
                ],
                "cache_info": cache_info
            }
        except Exception as e:
            logger.warning(f"FactCheckAgent 추가 정보 조회 실패: {str(e)}")
            return {"error": "추가 정보를 가져올 수 없습니다"}

    def get_info(self) -> Dict[str, Any]:
        """확장된 에이전트 정보"""
        base_info = super().get_info()
        base_info["agent_specific"] = self.get_agent_specific_info()
        return base_info


# 유틸리티 함수들

def validate_factcheck_config(config_dict: Dict[str, Any]) -> None:
    """팩트체킹 설정 유효성 검사"""
    if "max_search_results" in config_dict:
        max_results = config_dict["max_search_results"]
        if not isinstance(max_results, int) or max_results < 1 or max_results > 20:
            raise ValueError("max_search_results는 1-20 사이의 정수여야 합니다")
    
    if "search_timeout" in config_dict:
        timeout = config_dict["search_timeout"]
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("search_timeout은 양수여야 합니다")


def get_supported_fact_categories() -> List[str]:
    """지원하는 팩트체킹 카테고리 목록"""
    return ["정확성", "최신성", "출처 신뢰도"]


# 테스트용 함수
async def test_factcheck_wrapper():
    """FactCheckAgentWrapper 테스트"""
    print("FactCheckAgentWrapper 테스트 시작...")
    
    # Mock FactCheckAgent와 DocumentAgent 생성
    class MockFactCheckAgent:
        def __init__(self):
            self.model = "gpt-5-nano"
            self.max_search_results = 5
            self.search_timeout = 10
            self.serpapi_key = None
            self.search_cache = {}
            self.verification_cache = {}
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        
        async def analyze_document(self, document_agent, doc_id):
            # Mock 팩트체킹 이슈 생성
            from src.core.models import Issue, IssueType, TextLocation, generate_issue_id
            from datetime import datetime
            
            issue = Issue(
                issue_id=generate_issue_id(doc_id, "p001", TextLocation(start=0, end=10), IssueType.FACT),
                doc_id=doc_id,
                page_id="p001",
                issue_type=IssueType.FACT,
                text_location=TextLocation(start=0, end=10),
                bbox_location=None,
                element_id=None,
                original_text="테스트 사실 정보",
                message="Mock 팩트체킹 이슈 - 정보의 정확성을 확인하세요",
                suggestion="외부 출처로 검증하세요",
                confidence=0.75,
                confidence_level="medium",
                agent_name="fact_check_agent_test",
                detected_at=datetime.now()
            )
            return [issue]
        
        def get_factcheck_summary(self, issues):
            return {
                "total_fact_issues": len(issues),
                "accuracy_issues": len([i for i in issues if "정확성" in i.message]),
                "outdated_issues": len([i for i in issues if "최신성" in i.message]),
                "avg_confidence": sum(i.confidence for i in issues) / len(issues) if issues else 0,
                "recommendations": ["Mock 팩트체킹 권장사항"]
            }
    
    class MockDocumentAgent:
        def get_document(self, doc_id):
            from src.core.models import DocumentMeta
            return DocumentMeta(
                doc_id=doc_id,
                title="팩트체킹 테스트 문서",
                doc_type="pdf", 
                total_pages=3,
                file_path="factcheck_test.pdf"
            ) if doc_id == "test_doc" else None
    
    # Mock 에이전트로 래퍼 테스트
    mock_factcheck = MockFactCheckAgent()
    mock_document = MockDocumentAgent()
    wrapper = FactCheckAgentWrapper(mock_factcheck, mock_document)
    
    print(f"지원 액션: {wrapper.get_supported_actions()}")
    
    # analyze 액션 테스트
    request = {"action": "analyze", "doc_id": "test_doc"}
    result = await wrapper.handle(request)
    print(f"Analyze 테스트 - 성공: {result['success']}")
    if result['success']:
        print(f"   발견된 팩트체킹 이슈: {result['result']['total_issues']}개")
    
    # summary 액션 테스트 (캐시 사용)
    request = {"action": "summary", "doc_id": "test_doc"}
    result = await wrapper.handle(request)
    print(f"Summary 테스트 - 성공: {result['success']}")
    if result['success']:
        print(f"   캐시 사용: {result['result']['from_cache']}")
        print(f"   총 팩트체킹 이슈: {result['result']['summary']['total_fact_issues']}")
    
    # config 액션 테스트
    request = {"action": "config"}
    result = await wrapper.handle(request)
    print(f"Config 테스트 - 성공: {result['success']}")
    
    # 에이전트 정보 테스트
    info = wrapper.get_info()
    print(f"에이전트 정보: {info['name']}, 상태: {info['status']}")
    print(f"   분석 캐시: {info['agent_specific']['cache_info']['analysis_cache_count']}개")
    
    print("FactCheckAgentWrapper 테스트 완료!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_factcheck_wrapper())