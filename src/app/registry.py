# app/registry.py
"""
Agent Registry - 에이전트 등록 및 관리
"""

import logging
from typing import Dict, List, Optional
from src.app.agents.base import BaseAgentWrapper

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    에이전트 레지스트리 - 모든 에이전트 래퍼를 중앙에서 관리
    """
    
    def __init__(self):
        self._agents: Dict[str, BaseAgentWrapper] = {}
        logger.info("AgentRegistry 초기화")

    def register(self, name: str, agent: BaseAgentWrapper) -> None:
        """
        에이전트 등록
        
        Args:
            name: 에이전트 이름 (고유 식별자)
            agent: 에이전트 래퍼 인스턴스
        """
        if name in self._agents:
            logger.warning(f"에이전트 '{name}' 중복 등록 - 덮어씀")
        
        self._agents[name] = agent
        logger.info(f"에이전트 등록: {name} ({agent.__class__.__name__})")

    def get(self, name: str) -> BaseAgentWrapper:
        """
        에이전트 조회
        
        Args:
            name: 에이전트 이름
            
        Returns:
            BaseAgentWrapper: 에이전트 래퍼 인스턴스
            
        Raises:
            KeyError: 에이전트가 존재하지 않을 때
        """
        if name not in self._agents:
            available_agents = list(self._agents.keys())
            raise KeyError(f"에이전트 '{name}'을 찾을 수 없습니다. 사용 가능한 에이전트: {available_agents}")
        
        return self._agents[name]

    def exists(self, name: str) -> bool:
        """
        에이전트 존재 여부 확인
        
        Args:
            name: 에이전트 이름
            
        Returns:
            bool: 존재 여부
        """
        return name in self._agents

    def all(self) -> Dict[str, BaseAgentWrapper]:
        """
        모든 등록된 에이전트 반환
        
        Returns:
            Dict[str, BaseAgentWrapper]: 에이전트 이름 -> 래퍼 매핑
        """
        return dict(self._agents)

    def list_names(self) -> List[str]:
        """
        등록된 모든 에이전트 이름 목록
        
        Returns:
            List[str]: 에이전트 이름 목록
        """
        return list(self._agents.keys())

    def unregister(self, name: str) -> Optional[BaseAgentWrapper]:
        """
        에이전트 등록 해제 (주로 테스트용)
        
        Args:
            name: 에이전트 이름
            
        Returns:
            Optional[BaseAgentWrapper]: 제거된 에이전트 (없으면 None)
        """
        if name not in self._agents:
            logger.warning(f"에이전트 '{name}' 등록 해제 실패 - 존재하지 않음")
            return None
        
        agent = self._agents.pop(name)
        logger.info(f"에이전트 등록 해제: {name}")
        return agent

    def clear(self) -> None:
        """
        모든 에이전트 등록 해제 (주로 테스트용)
        """
        count = len(self._agents)
        self._agents.clear()
        logger.info(f"모든 에이전트 등록 해제: {count}개")

    def get_stats(self) -> Dict[str, any]:
        """
        레지스트리 통계 정보
        
        Returns:
            Dict: 통계 정보
        """
        return {
            "total_agents": len(self._agents),
            "agent_names": self.list_names(),
            "agent_types": [agent.__class__.__name__ for agent in self._agents.values()]
        }

    def __len__(self) -> int:
        """등록된 에이전트 수"""
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        """에이전트 존재 여부 (in 연산자 지원)"""
        return name in self._agents

    def __repr__(self) -> str:
        """레지스트리 문자열 표현"""
        return f"AgentRegistry(agents={list(self._agents.keys())})"