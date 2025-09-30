# app/wrappers/document_wrapper.py
"""
DocumentAgent Wrapper - 문서 처리 에이전트 래퍼
"""

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from src.app.wrappers.base import BaseAgentWrapper, validate_doc_id
from src.agents.document_agent import DocumentAgent
from src.core.models import DocumentMeta

logger = logging.getLogger(__name__)


class DocumentAgentWrapper(BaseAgentWrapper):
    """
    DocumentAgent를 위한 래퍼 클래스
    
    지원하는 액션:
    - process: 문서 파일 처리 (파일 경로 → DocumentMeta)
    - stats: 문서 통계 정보 조회
    - search: 문서 내 의미적 검색
    - info: 문서 메타데이터 조회
    - list: 처리된 모든 문서 목록
    - slide: 특정 슬라이드 데이터 조회
    """
    
    def __init__(self, document_agent: DocumentAgent):
        super().__init__("document", document_agent)
        self.document_agent = document_agent
        logger.info("DocumentAgentWrapper 초기화 완료")

    def get_supported_actions(self) -> List[str]:
        """지원하는 액션 목록"""
        return ["process", "stats", "search", "info", "list", "slide"]

    async def _execute_action(self, action: str, request: Dict[str, Any]) -> Any:
        """실제 액션 실행 로직"""
        
        if action == "process":
            return await self._process_document(request)
        elif action == "stats":
            return await self._get_stats(request)
        elif action == "search":
            return await self._search_document(request)
        elif action == "info":
            return await self._get_info(request)
        elif action == "list":
            return await self._list_documents(request)
        elif action == "slide":
            return await self._get_slide(request)
        else:
            raise ValueError(f"지원하지 않는 액션: {action}")

    async def _process_document(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        문서 처리 액션
        
        Args:
            request: {"file_path": "path/to/document.pdf"}
            
        Returns:
            Dict: 처리된 문서 메타데이터
        """
        file_path = request.get("file_path")
        if not file_path:
            raise ValueError("file_path가 필요합니다")
        
        # 파일 존재 확인
        if not Path(file_path).exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        # 지원 형식 확인
        supported_extensions = [".pdf"]  # PPT는 현재 비활성화
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in supported_extensions:
            raise ValueError(f"지원하지 않는 파일 형식: {file_ext}. 지원 형식: {supported_extensions}")
        
        logger.info(f"문서 처리 시작: {file_path}")
        
        try:
            # DocumentAgent로 실제 처리
            doc_meta = await self.document_agent.process_document(file_path)
            
            logger.info(f"문서 처리 완료: {doc_meta.doc_id} ({doc_meta.total_pages} 페이지)")
            
            return {
                "doc_meta": doc_meta.model_dump(mode='json'),  # JSON 호환 직렬화
                "message": f"문서 처리 완료: {doc_meta.total_pages} 페이지"
            }
            
        except Exception as e:
            logger.error(f"문서 처리 실패: {str(e)}")
            raise RuntimeError(f"문서 처리 중 오류 발생: {str(e)}")

    async def _get_stats(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        문서 통계 조회 액션
        
        Args:
            request: {"doc_id": "document_id"}
            
        Returns:
            Dict: 문서 통계 정보
        """
        doc_id = validate_doc_id(request.get("doc_id"))
        
        # 문서 존재 확인
        if not self.document_agent.get_document(doc_id):
            raise ValueError(f"문서를 찾을 수 없습니다: {doc_id}")
        
        try:
            stats = self.document_agent.get_document_stats(doc_id)
            
            if not stats:
                raise ValueError(f"문서 통계를 가져올 수 없습니다: {doc_id}")
            
            return {
                "doc_id": doc_id,
                "stats": stats,
                "message": "문서 통계 조회 완료"
            }
            
        except Exception as e:
            logger.error(f"문서 통계 조회 실패: {str(e)}")
            raise RuntimeError(f"문서 통계 조회 중 오류 발생: {str(e)}")

    async def _search_document(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        문서 내 검색 액션
        
        Args:
            request: {
                "doc_id": "document_id",
                "query": "검색어",
                "top_k": 5  # 선택사항
            }
            
        Returns:
            Dict: 검색 결과 목록
        """
        doc_id = validate_doc_id(request.get("doc_id"))
        query = request.get("query")
        
        if not query or not isinstance(query, str) or len(query.strip()) == 0:
            raise ValueError("검색어(query)가 필요합니다")
        
        top_k = request.get("top_k", 5)
        if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
            raise ValueError("top_k는 1-20 사이의 정수여야 합니다")
        
        # 문서 존재 확인
        if not self.document_agent.get_document(doc_id):
            raise ValueError(f"문서를 찾을 수 없습니다: {doc_id}")
        
        try:
            results = self.document_agent.search_in_document(doc_id, query.strip(), top_k)
            
            return {
                "doc_id": doc_id,
                "query": query,
                "results": results,
                "total_results": len(results),
                "message": f"검색 완료: {len(results)}개 결과"
            }
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {str(e)}")
            raise RuntimeError(f"문서 검색 중 오류 발생: {str(e)}")

    async def _get_info(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        문서 메타데이터 조회 액션
        
        Args:
            request: {"doc_id": "document_id"}
            
        Returns:
            Dict: 문서 메타데이터
        """
        doc_id = validate_doc_id(request.get("doc_id"))
        
        try:
            doc_meta = self.document_agent.get_document(doc_id)
            
            if not doc_meta:
                raise ValueError(f"문서를 찾을 수 없습니다: {doc_id}")
            
            return {
                "doc_id": doc_id,
                "doc_meta": doc_meta.model_dump(mode='json'),  # JSON 호환 직렬화
                "message": "문서 정보 조회 완료"
            }
            
        except Exception as e:
            logger.error(f"문서 정보 조회 실패: {str(e)}")
            raise RuntimeError(f"문서 정보 조회 중 오류 발생: {str(e)}")

    async def _list_documents(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        처리된 모든 문서 목록 조회 액션
        
        Args:
            request: {} (파라미터 없음)
            
        Returns:
            Dict: 문서 목록
        """
        try:
            documents = self.document_agent.list_documents()
            
            return {
                "documents": [doc.model_dump(mode='json') for doc in documents],  # JSON 호환 직렬화
                "total_documents": len(documents),
                "message": f"문서 목록 조회 완료: {len(documents)}개 문서"
            }
            
        except Exception as e:
            logger.error(f"문서 목록 조회 실패: {str(e)}")
            raise RuntimeError(f"문서 목록 조회 중 오류 발생: {str(e)}")

    async def _get_slide(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        특정 슬라이드 데이터 조회 액션
        
        Args:
            request: {
                "doc_id": "document_id",
                "page_id": "p001"  # 선택사항, 없으면 모든 슬라이드
            }
            
        Returns:
            Dict: 슬라이드 데이터
        """
        doc_id = validate_doc_id(request.get("doc_id"))
        page_id = request.get("page_id")
        
        # 문서 존재 확인
        if not self.document_agent.get_document(doc_id):
            raise ValueError(f"문서를 찾을 수 없습니다: {doc_id}")
        
        try:
            if page_id:
                # 특정 슬라이드 조회
                slide_data = self.document_agent.get_slide_by_page_id(doc_id, page_id)
                
                if not slide_data:
                    raise ValueError(f"슬라이드를 찾을 수 없습니다: {page_id}")
                
                return {
                    "doc_id": doc_id,
                    "page_id": page_id,
                    "slide_data": slide_data,
                    "message": f"슬라이드 조회 완료: {page_id}"
                }
            else:
                # 모든 슬라이드 조회
                all_slides = self.document_agent.get_slide_data(doc_id)
                
                # 이미지 데이터 제외하고 메타데이터만 반환 (응답 크기 최적화)
                slides_meta = []
                for slide in all_slides:
                    slide_meta = {
                        "doc_id": slide.get("doc_id"),
                        "page_id": slide.get("page_id"),
                        "page_number": slide.get("page_number"),
                        "caption": slide.get("caption"),
                        "dimensions": slide.get("dimensions"),
                        "size_bytes": slide.get("size_bytes"),
                        "has_image": bool(slide.get("image_base64"))
                    }
                    slides_meta.append(slide_meta)
                
                return {
                    "doc_id": doc_id,
                    "slides": slides_meta,
                    "total_slides": len(slides_meta),
                    "message": f"전체 슬라이드 조회 완료: {len(slides_meta)}개"
                }
                
        except Exception as e:
            logger.error(f"슬라이드 조회 실패: {str(e)}")
            raise RuntimeError(f"슬라이드 조회 중 오류 발생: {str(e)}")

    def _validate_request(self, request: Dict[str, Any]) -> None:
        """DocumentAgent 전용 요청 유효성 검사"""
        super()._validate_request(request)
        
        action = request["action"]
        
        # 액션별 추가 유효성 검사
        if action == "process":
            if "file_path" not in request:
                raise ValueError("process 액션은 file_path가 필요합니다")
        
        elif action in ["stats", "search", "info", "slide"]:
            if "doc_id" not in request:
                raise ValueError(f"{action} 액션은 doc_id가 필요합니다")
        
        elif action == "search":
            if "query" not in request:
                raise ValueError("search 액션은 query가 필요합니다")

    def get_agent_specific_info(self) -> Dict[str, Any]:
        """DocumentAgent 관련 추가 정보"""
        try:
            # 처리된 문서 수
            documents = self.document_agent.list_documents()
            doc_count = len(documents)
            
            # 총 슬라이드 수 계산
            total_slides = sum(doc.total_pages for doc in documents)
            
            return {
                "processed_documents": doc_count,
                "total_slides": total_slides,
                "avg_slides_per_doc": round(total_slides / doc_count, 1) if doc_count > 0 else 0,
                "recent_documents": [
                    {"doc_id": doc.doc_id, "title": doc.title, "pages": doc.total_pages}
                    for doc in documents[-3:]  # 최근 3개
                ]
            }
        except Exception as e:
            logger.warning(f"DocumentAgent 추가 정보 조회 실패: {str(e)}")
            return {"error": "추가 정보를 가져올 수 없습니다"}

    def get_info(self) -> Dict[str, Any]:
        """확장된 에이전트 정보"""
        base_info = super().get_info()
        base_info["agent_specific"] = self.get_agent_specific_info()
        return base_info


# 유틸리티 함수들

def validate_file_path(file_path: str) -> Path:
    """파일 경로 유효성 검사"""
    if not file_path or not isinstance(file_path, str):
        raise ValueError("file_path는 비어있지 않은 문자열이어야 합니다")
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    if not path.is_file():
        raise ValueError(f"파일이 아닙니다: {file_path}")
    
    return path


def get_supported_file_extensions() -> List[str]:
    """지원하는 파일 확장자 목록"""
    return [".pdf"]


# 테스트용 함수
async def test_document_wrapper():
    """DocumentAgentWrapper 테스트"""
    print("DocumentAgentWrapper 테스트 시작...")
    
    # Mock DocumentAgent 생성 (실제 환경에서는 제거)
    class MockDocumentAgent:
        def __init__(self):
            self.docs = {}
        
        async def process_document(self, file_path):
            from src.core.models import DocumentMeta, generate_doc_id
            doc_id = generate_doc_id(file_path)
            meta = DocumentMeta(
                doc_id=doc_id,
                title="테스트 문서",
                doc_type="pdf",
                total_pages=10,
                file_path=file_path
            )
            self.docs[doc_id] = meta
            return meta
        
        def get_document(self, doc_id):
            return self.docs.get(doc_id)
        
        def list_documents(self):
            return list(self.docs.values())
        
        def get_document_stats(self, doc_id):
            return {"doc_id": doc_id, "total_slides": 10, "caption_coverage": 1.0}
    
    # Mock 에이전트로 테스트
    mock_agent = MockDocumentAgent()
    wrapper = DocumentAgentWrapper(mock_agent)
    
    print(f"지원 액션: {wrapper.get_supported_actions()}")
    
    # list 액션 테스트
    request = {"action": "list"}
    result = await wrapper.handle(request)
    print(f"List 테스트 - 성공: {result['success']}")
    
    # 에이전트 정보 테스트
    info = wrapper.get_info()
    print(f"에이전트 정보: {info['name']}, 통계: {info['stats']['total_requests']}")
    
    print("DocumentAgentWrapper 테스트 완료!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_document_wrapper())