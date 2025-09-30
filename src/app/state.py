# app/state.py
"""
Application State - 전역 상태 관리
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from src.app.registry import AgentRegistry

logger = logging.getLogger(__name__)


class AppState:
    """
    애플리케이션 전역 상태 관리 클래스
    - Agent Registry 관리
    - Executor/Aggregator 지연 로딩
    - 애플리케이션 메타데이터 저장
    """
    
    def __init__(self):
        # 핵심 컴포넌트들
        self.registry = AgentRegistry()
        
        # 지연 로딩될 컴포넌트들 (순환 참조 방지)
        self._executor = None
        self._aggregator = None
        
        # 애플리케이션 메타데이터
        self.startup_time = datetime.now()
        self.version = "0.1.0"
        self.debug = False
        
        # 실행 통계
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "documents_processed": 0,
            "analyses_performed": 0
        }
        
        # 런타임 설정 (환경변수 오버라이드 가능)
        self.config = {
            "max_file_size_mb": 50,
            "upload_timeout_seconds": 300,
            "analysis_timeout_seconds": 600,
            "max_concurrent_analyses": 3,
        }
        
        logger.info("AppState 초기화 완료")

    def increment_stat(self, stat_name: str, value: int = 1) -> None:
        """
        통계 카운터 증가
        
        Args:
            stat_name: 통계 이름
            value: 증가할 값 (기본 1)
        """
        if stat_name in self._stats:
            self._stats[stat_name] += value
        else:
            logger.warning(f"알 수 없는 통계 이름: {stat_name}")

    def get_stats(self) -> Dict[str, Any]:
        """
        현재 애플리케이션 통계 반환
        
        Returns:
            Dict: 통계 정보
        """
        uptime = (datetime.now() - self.startup_time).total_seconds()
        
        return {
            "version": self.version,
            "uptime_seconds": uptime,
            "startup_time": self.startup_time.isoformat(),
            "registry_stats": self.registry.get_stats(),
            "execution_stats": dict(self._stats),
            "config": dict(self.config),
        }

    def update_config(self, key: str, value: Any) -> None:
        """
        런타임 설정 업데이트
        
        Args:
            key: 설정 키
            value: 설정 값
        """
        old_value = self.config.get(key)
        self.config[key] = value
        logger.info(f"설정 업데이트: {key} = {value} (이전: {old_value})")

    def is_ready(self) -> bool:
        """
        애플리케이션이 요청을 처리할 준비가 되었는지 확인
        
        Returns:
            bool: 준비 상태
        """
        # 최소한 하나의 에이전트가 등록되어 있어야 함
        if len(self.registry) == 0:
            return False
        
        # 필수 에이전트들이 등록되어 있는지 확인
        required_agents = ["document", "quality"]  # fact는 선택사항
        for agent_name in required_agents:
            if not self.registry.exists(agent_name):
                logger.warning(f"필수 에이전트 누락: {agent_name}")
                return False
        
        return True

    def reset_stats(self) -> None:
        """통계 초기화 (주로 테스트용)"""
        for key in self._stats:
            self._stats[key] = 0
        logger.info("통계 초기화 완료")

    async def shutdown(self) -> None:
        """애플리케이션 종료 시 정리 작업"""
        logger.info("AppState 종료 프로세스 시작")
        
        # 실행 중인 작업들 정리 (필요시)
        # await self._cleanup_background_tasks()
        
        # 통계 로깅
        final_stats = self.get_stats()
        logger.info(f"최종 통계: {final_stats['execution_stats']}")
        
        logger.info("AppState 종료 완료")

    def __repr__(self) -> str:
        """AppState 문자열 표현"""
        ready = "Ready" if self.is_ready() else "Not Ready"
        return f"AppState(version={self.version}, agents={len(self.registry)}, status={ready})"


# 전역 애플리케이션 상태 인스턴스 (싱글톤 패턴)
app_state = AppState()


# FastAPI 앱에서 사용할 의존성 함수들
def get_app_state() -> AppState:
    """
    FastAPI 의존성으로 사용할 AppState 인스턴스 반환
    
    Returns:
        AppState: 전역 상태 인스턴스
    """
    return app_state


def get_registry() -> AgentRegistry:
    """
    FastAPI 의존성으로 사용할 AgentRegistry 인스턴스 반환
    
    Returns:
        AgentRegistry: 에이전트 레지스트리
    """
    return app_state.registry


# 애플리케이션 초기화 헬퍼 함수
async def initialize_app() -> None:
    """
    애플리케이션 초기화 - main.py에서 호출
    """
    logger.info("애플리케이션 초기화 시작")
    
    # 환경변수 기반 설정 로드 (선택사항)
    import os
    
    # 디버그 모드 설정
    app_state.debug = os.getenv("DEBUG", "false").lower() == "true"
    
    # 설정 오버라이드
    if max_size := os.getenv("MAX_FILE_SIZE_MB"):
        app_state.update_config("max_file_size_mb", int(max_size))
    
    logger.info(f"애플리케이션 초기화 완료: {app_state}")


# 애플리케이션 종료 헬퍼 함수  
async def shutdown_app() -> None:
    """
    애플리케이션 종료 - main.py에서 호출
    """
    await app_state.shutdown()