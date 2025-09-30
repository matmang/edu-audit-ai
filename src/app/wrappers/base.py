# app/agents/base.py
"""
Base Agent Wrapper - 모든 에이전트 래퍼의 공통 인터페이스
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """에이전트 상태"""
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"


class BaseAgentWrapper(ABC):
    """
    모든 에이전트 래퍼의 기본 클래스
    
    모든 에이전트는 이 인터페이스를 구현하여 일관된 호출 방식을 제공
    """
    
    def __init__(self, name: str, agent_instance: Any = None):
        self.name = name
        self._agent = agent_instance
        self._status = AgentStatus.READY
        self._last_error: Optional[str] = None
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0
        }
        
        logger.info(f"에이전트 래퍼 초기화: {name}")

    @property
    def status(self) -> AgentStatus:
        """현재 에이전트 상태"""
        return self._status

    @property
    def last_error(self) -> Optional[str]:
        """마지막 에러 메시지"""
        return self._last_error

    @property
    def stats(self) -> Dict[str, Union[int, float]]:
        """에이전트 실행 통계"""
        return dict(self._stats)

    @abstractmethod
    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        에이전트 요청 처리 메인 메서드
        
        Args:
            request: 요청 데이터
                - action: 수행할 작업 (analyze, summary, search 등)
                - doc_id: 대상 문서 ID (선택사항)
                - 기타 에이전트별 파라미터들
                
        Returns:
            Dict: 응답 데이터
                - agent: 에이전트 이름
                - action: 수행된 작업
                - success: 성공 여부
                - result: 결과 데이터 (에이전트별로 상이)
                - error: 에러 메시지 (실패 시)
                - processing_time: 처리 시간 (초)
        """
        pass

    def get_supported_actions(self) -> List[str]:
        """
        지원하는 액션 목록 반환
        하위 클래스에서 오버라이드 권장
        
        Returns:
            List[str]: 지원하는 액션 이름들
        """
        return ["analyze", "summary"]

    async def _execute_with_stats(self, action: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        통계 추적과 함께 실제 작업 실행
        
        Args:
            action: 실행할 액션
            request: 요청 데이터
            
        Returns:
            Dict: 표준화된 응답
        """
        self._stats["total_requests"] += 1
        start_time = time.time()
        
        try:
            self._status = AgentStatus.BUSY
            
            # 실제 작업 실행
            result = await self._execute_action(action, request)
            
            # 성공 처리
            self._status = AgentStatus.READY
            self._last_error = None
            self._stats["successful_requests"] += 1
            
            processing_time = time.time() - start_time
            self._update_timing_stats(processing_time)
            
            return {
                "agent": self.name,
                "action": action,
                "success": True,
                "result": result,
                "processing_time": processing_time
            }
            
        except Exception as e:
            # 에러 처리
            self._status = AgentStatus.ERROR
            self._last_error = str(e)
            self._stats["failed_requests"] += 1
            
            processing_time = time.time() - start_time
            self._update_timing_stats(processing_time)
            
            logger.error(f"에이전트 {self.name} 실행 실패: {str(e)}")
            
            return {
                "agent": self.name,
                "action": action,
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }

    @abstractmethod
    async def _execute_action(self, action: str, request: Dict[str, Any]) -> Any:
        """
        실제 액션 실행 로직 (하위 클래스에서 구현)
        
        Args:
            action: 실행할 액션
            request: 요청 데이터
            
        Returns:
            Any: 액션별 결과 데이터
        """
        pass

    def _update_timing_stats(self, processing_time: float) -> None:
        """처리 시간 통계 업데이트"""
        self._stats["total_processing_time"] += processing_time
        
        total_requests = self._stats["total_requests"]
        if total_requests > 0:
            self._stats["avg_processing_time"] = (
                self._stats["total_processing_time"] / total_requests
            )

    def _validate_request(self, request: Dict[str, Any]) -> None:
        """
        요청 유효성 검사 (하위 클래스에서 오버라이드 가능)
        
        Args:
            request: 요청 데이터
            
        Raises:
            ValueError: 요청이 유효하지 않을 때
        """
        if not isinstance(request, dict):
            raise ValueError("요청은 딕셔너리여야 합니다")
        
        action = request.get("action")
        if not action:
            raise ValueError("action 필드가 필요합니다")
        
        supported_actions = self.get_supported_actions()
        if action not in supported_actions:
            raise ValueError(f"지원하지 않는 액션입니다: {action}. 지원하는 액션: {supported_actions}")

    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        공통 요청 처리 템플릿 메서드
        """
        try:
            # 요청 유효성 검사
            self._validate_request(request)
            
            action = request["action"]
            
            # 에이전트 상태 체크
            if self._status == AgentStatus.DISABLED:
                return {
                    "agent": self.name,
                    "action": action,
                    "success": False,
                    "error": "에이전트가 비활성화되었습니다"
                }
            
            # 통계와 함께 실행
            return await self._execute_with_stats(action, request)
            
        except Exception as e:
            print(e)
            logger.error(f"에이전트 {self.name} 요청 처리 실패: {str(e)}")
            return {
                "agent": self.name,
                "action": request.get("action", "unknown"),
                "success": False,
                "error": f"요청 처리 실패: {str(e)}"
            }

    def reset_stats(self) -> None:
        """통계 초기화"""
        for key in self._stats:
            if isinstance(self._stats[key], (int, float)):
                self._stats[key] = 0 if isinstance(self._stats[key], int) else 0.0
        logger.info(f"에이전트 {self.name} 통계 초기화")

    def set_status(self, status: AgentStatus) -> None:
        """에이전트 상태 수동 설정 (주로 관리용)"""
        old_status = self._status
        self._status = status
        logger.info(f"에이전트 {self.name} 상태 변경: {old_status} -> {status}")

    def get_info(self) -> Dict[str, Any]:
        """에이전트 정보 반환"""
        return {
            "name": self.name,
            "status": self._status.value,
            "supported_actions": self.get_supported_actions(),
            "last_error": self._last_error,
            "stats": self.stats,
            "agent_type": self.__class__.__name__
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, status={self._status.value})"


class MockAgentWrapper(BaseAgentWrapper):
    """
    테스트용 Mock 에이전트 래퍼
    """
    
    def __init__(self, name: str = "mock", delay: float = 0.1):
        super().__init__(name)
        self.delay = delay

    def get_supported_actions(self) -> List[str]:
        return ["analyze", "summary", "test"]

    async def _execute_action(self, action: str, request: Dict[str, Any]) -> Any:
        """Mock 에이전트의 더미 실행 로직"""
        import asyncio
        
        # 인위적 지연
        await asyncio.sleep(self.delay)
        
        doc_id = request.get("doc_id", "unknown")
        
        if action == "analyze":
            return {
                "doc_id": doc_id,
                "issues": [
                    {
                        "issue_type": "mock",
                        "message": f"Mock 이슈 from {self.name}",
                        "confidence": 0.8
                    }
                ],
                "total_issues": 1
            }
        elif action == "summary":
            return {
                "doc_id": doc_id,
                "summary": {
                    "total_issues": 1,
                    "recommendations": [f"Mock 권장사항 from {self.name}"]
                }
            }
        elif action == "test":
            return {
                "message": f"테스트 액션 실행됨 - {self.name}",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise ValueError(f"알 수 없는 액션: {action}")


# 유틸리티 함수들

def create_error_response(agent_name: str, action: str, error_msg: str) -> Dict[str, Any]:
    """표준 에러 응답 생성"""
    return {
        "agent": agent_name,
        "action": action,
        "success": False,
        "error": error_msg,
        "processing_time": 0.0
    }


def validate_doc_id(doc_id: Optional[str]) -> str:
    """doc_id 유효성 검사"""
    if not doc_id:
        raise ValueError("doc_id가 필요합니다")
    if not isinstance(doc_id, str) or len(doc_id.strip()) == 0:
        raise ValueError("doc_id는 비어있지 않은 문자열이어야 합니다")
    return doc_id.strip()


# 테스트용 함수
async def test_base_wrapper():
    """BaseAgentWrapper 기본 테스트"""
    print("BaseAgentWrapper 테스트 시작...")
    
    # Mock 에이전트 생성
    mock_agent = MockAgentWrapper("test_mock", delay=0.1)
    
    # 지원 액션 테스트
    actions = mock_agent.get_supported_actions()
    print(f"지원 액션: {actions}")
    
    # 정상 요청 테스트
    request = {"action": "analyze", "doc_id": "test_doc"}
    result = await mock_agent.handle(request)
    print(f"정상 요청 결과: {result['success']}")
    
    # 에러 요청 테스트
    bad_request = {"action": "unknown"}
    result = await mock_agent.handle(bad_request)
    print(f"에러 요청 결과: {result['success']}")
    
    # 통계 확인
    stats = mock_agent.stats
    print(f"통계: 총 요청 {stats['total_requests']}, 성공 {stats['successful_requests']}")
    
    # 에이전트 정보
    info = mock_agent.get_info()
    print(f"에이전트 정보: {info['name']}, 상태: {info['status']}")
    
    print("테스트 완료!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_base_wrapper())