# app/main.py
"""
EDU-Audit FastAPI Main Server
DocumentAgent 테스트를 위한 기본 서버 구현
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# 앱 상태 및 컴포넌트들
from src.app.state import app_state, initialize_app, shutdown_app, get_app_state
from src.app.agents.document_wrapper import DocumentAgentWrapper
from src.app.agents.quality_wrapper import QualityAgentWrapper
from src.app.agents.factcheck_wrapper import FactCheckAgentWrapper

# 실제 에이전트 임포트
try:
    from src.agents.document_agent import DocumentAgent
    from src.agents.quality_agent import QualityAgent, QualityConfig
    from src.agents.factcheck_agent import FactCheckAgent
    AGENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"에이전트 임포트 실패: {str(e)}")
    AGENTS_AVAILABLE = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    logger.info("EDU-Audit 서버 시작")
    
    # 시작 시 초기화
    await initialize_app()
    await setup_agents()
    
    yield
    
    # 종료 시 정리
    await shutdown_app()
    logger.info("EDU-Audit 서버 종료")


# FastAPI 앱 생성
app = FastAPI(
    title="EDU-Audit MCP Server",
    description="교육 자료 품질 검수 서비스",
    version="0.1.0",
    lifespan=lifespan
)

# CORS 설정 (개발용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def setup_agents():
    """에이전트 초기화 및 등록"""
    logger.info("에이전트 설정 시작")
    
    if not AGENTS_AVAILABLE:
        logger.error("에이전트를 임포트할 수 없습니다. src 모듈 경로를 확인하세요.")
        return
    
    try:
        # 환경변수에서 API 키 읽기
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            return
        
        # DocumentAgent 생성
        document_agent = DocumentAgent(
            openai_api_key=openai_api_key,
            vision_model=os.getenv("VISION_MODEL", "gpt-5-nano"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        )
        
        # Wrapper로 감싸서 등록
        document_wrapper = DocumentAgentWrapper(document_agent)
        app_state.registry.register("document", document_wrapper)
        
        # QualityAgent 생성
        quality_agent = QualityAgent(
            openai_api_key=openai_api_key,
            vision_model=os.getenv("VISION_MODEL", "gpt-5-nano"),
            config=QualityConfig()
        )
        
        quality_wrapper = QualityAgentWrapper(quality_agent, document_agent)
        app_state.registry.register("quality", quality_wrapper)

        logger.info("QualityAgent 등록 완료")

        # FactCheckAgent 생성
        factcheck_agent = FactCheckAgent(
            openai_api_key=openai_api_key,
            serpapi_key=os.getenv("SERPAPI_API_KEY"),
            model=os.getenv("LLM_MODEL", "gpt-5-nano")
        )

        factcheck_wrapper = FactCheckAgentWrapper(factcheck_agent, document_agent)
        app_state.registry.register("factcheck", factcheck_wrapper)
        
        logger.info("FactCheckAgent 등록 완료")
        
    except Exception as e:
        logger.error(f"에이전트 설정 실패: {str(e)}")
        raise


# ── 헬스 체크 엔드포인트 ──

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "ok",
        "version": app_state.version,
        "ready": app_state.is_ready(),
        "agents": app_state.registry.list_names()
    }

@app.get("/stats")
async def get_stats(state: Any = Depends(get_app_state)):
    """서버 통계 정보"""
    return JSONResponse(state.get_stats())


# ── 문서 처리 엔드포인트 ──

@app.post("/document/upload")
async def upload_document(
    file: UploadFile = File(...),
    state: Any = Depends(get_app_state)
):
    """
    문서 파일 업로드 및 처리
    
    지원 형식: PDF
    최대 크기: 50MB (설정에서 변경 가능)
    """
    try:
        # 파일 크기 체크
        max_size_mb = state.config["max_file_size_mb"]
        if file.size and file.size > max_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"파일 크기가 너무 큽니다. 최대 {max_size_mb}MB까지 허용됩니다."
            )
        
        # 파일 확장자 체크
        if not file.filename:
            raise HTTPException(status_code=400, detail="파일명이 필요합니다.")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in [".pdf"]:
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 파일 형식: {file_ext}. PDF만 지원됩니다."
            )
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        logger.info(f"파일 업로드 완료: {file.filename} -> {tmp_path}")
        
        try:
            # DocumentAgent를 통해 처리
            document_wrapper = state.registry.get("document")
            
            request = {
                "action": "process",
                "file_path": tmp_path
            }
            
            result = await document_wrapper.handle(request)
            
            # 통계 업데이트
            if result["success"]:
                state.increment_stat("documents_processed")
                state.increment_stat("successful_requests")
                
                # DocumentMeta의 datetime 필드를 JSON 직렬화 가능하게 변환
                doc_meta_dict = result["result"]["doc_meta"]
                
                return JSONResponse({
                    "success": True,
                    "message": "문서 처리 완료",
                    "doc_meta": doc_meta_dict,
                    "processing_time": result["processing_time"]
                })
            else:
                state.increment_stat("failed_requests")
                raise HTTPException(
                    status_code=500,
                    detail=f"문서 처리 실패: {result['error']}"
                )
                
        finally:
            # 임시 파일 정리
            try:
                Path(tmp_path).unlink(missing_ok=True)
                logger.info(f"임시 파일 삭제: {tmp_path}")
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        state.increment_stat("failed_requests")
        logger.error(f"문서 업로드 처리 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")


# ── 문서 조회 엔드포인트들 ──

@app.get("/document/list")
async def list_documents(state: Any = Depends(get_app_state)):
    """처리된 모든 문서 목록 조회"""
    try:
        document_wrapper = state.registry.get("document")
        
        request = {"action": "list"}
        result = await document_wrapper.handle(request)
        
        if result["success"]:
            return JSONResponse(result["result"])
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except KeyError:
        raise HTTPException(status_code=503, detail="DocumentAgent가 등록되지 않았습니다.")
    except Exception as e:
        logger.error(f"문서 목록 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ── 품질 분석 엔드포인트들 ──

@app.post("/document/{doc_id}/analyze/quality")
async def analyze_quality(doc_id: str, state: Any = Depends(get_app_state)):
    """문서 품질 분석 (오탈자, 문법, 가독성 등)"""
    try:
        quality_wrapper = state.registry.get("quality")
        
        request = {
            "action": "analyze",
            "doc_id": doc_id
        }
        result = await quality_wrapper.handle(request)
        
        # 통계 업데이트
        state.increment_stat("analyses_performed")
        
        if result["success"]:
            state.increment_stat("successful_requests")
            return JSONResponse(result["result"])
        else:
            state.increment_stat("failed_requests")
            if "찾을 수 없습니다" in result["error"]:
                raise HTTPException(status_code=404, detail=result["error"])
            raise HTTPException(status_code=500, detail=result["error"])
            
    except KeyError:
        raise HTTPException(status_code=503, detail="QualityAgent가 등록되지 않았습니다.")
    except HTTPException:
        raise
    except Exception as e:
        state.increment_stat("failed_requests")
        logger.error(f"품질 분석 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document/{doc_id}/summary/quality")
async def get_quality_summary(doc_id: str, state: Any = Depends(get_app_state)):
    """문서 품질 분석 요약"""
    try:
        quality_wrapper = state.registry.get("quality")
        
        request = {
            "action": "summary",
            "doc_id": doc_id
        }
        result = await quality_wrapper.handle(request)
        
        if result["success"]:
            return JSONResponse(result["result"])
        else:
            if "찾을 수 없습니다" in result["error"]:
                raise HTTPException(status_code=404, detail=result["error"])
            raise HTTPException(status_code=500, detail=result["error"])
            
    except KeyError:
        raise HTTPException(status_code=503, detail="QualityAgent가 등록되지 않았습니다.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"품질 요약 생성 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quality/config")
async def get_quality_config(state: Any = Depends(get_app_state)):
    """QualityAgent 설정 조회"""
    try:
        quality_wrapper = state.registry.get("quality")
        
        request = {"action": "config"}
        result = await quality_wrapper.handle(request)
        
        if result["success"]:
            return JSONResponse(result["result"])
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except KeyError:
        raise HTTPException(status_code=503, detail="QualityAgent가 등록되지 않았습니다.")
    except Exception as e:
        logger.error(f"QualityAgent 설정 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/document/{doc_id}/cache/quality")
async def clear_quality_cache(doc_id: str, state = Depends(get_app_state)):
    """특정 문서의 품질 분석 캐시 삭제"""
    quality_wrapper = state.registry.get("quality") 
    result = await quality_wrapper.handle({"action": "clear_cache", "doc_id": doc_id})
    return JSONResponse(result)

# ── 팩트체킹 분석 엔드포인트 ──

@app.post("/document/{doc_id}/analyze/factcheck")
async def analyze_factcheck(
    doc_id: str, 
    request_body: Optional[Dict[str, Any]] = None,
    state: Any = Depends(get_app_state)
):
    """문서 팩트체킹 분석 (선택적 검색 기반)"""
    try:
        factcheck_wrapper = state.registry.get("factcheck")
        
        # 요청 본문에서 옵션 추출
        options = request_body or {}
        force_reanalyze = options.get("force_reanalyze", False)
        
        request = {
            "action": "analyze",
            "doc_id": doc_id,
            "force_reanalyze": force_reanalyze
        }
        result = await factcheck_wrapper.handle(request)
        
        # 통계 업데이트
        state.increment_stat("analyses_performed")
        
        if result["success"]:
            state.increment_stat("successful_requests")
            return JSONResponse(result["result"])
        else:
            state.increment_stat("failed_requests")
            if "찾을 수 없습니다" in result["error"]:
                raise HTTPException(status_code=404, detail=result["error"])
            raise HTTPException(status_code=500, detail=result["error"])
            
    except KeyError:
        raise HTTPException(status_code=503, detail="FactCheckAgent가 등록되지 않았습니다.")
    except HTTPException:
        raise
    except Exception as e:
        state.increment_stat("failed_requests")
        logger.error(f"팩트체킹 분석 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document/{doc_id}/summary/factcheck")
async def get_factcheck_summary(doc_id: str, state: Any = Depends(get_app_state)):
    """문서 팩트체킹 분석 요약"""
    try:
        factcheck_wrapper = state.registry.get("factcheck")
        
        request = {
            "action": "summary",
            "doc_id": doc_id
        }
        result = await factcheck_wrapper.handle(request)
        
        if result["success"]:
            return JSONResponse(result["result"])
        else:
            if "찾을 수 없습니다" in result["error"]:
                raise HTTPException(status_code=404, detail=result["error"])
            raise HTTPException(status_code=500, detail=result["error"])
            
    except KeyError:
        raise HTTPException(status_code=503, detail="FactCheckAgent가 등록되지 않았습니다.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"팩트체킹 요약 생성 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/factcheck/config")
async def get_factcheck_config(state: Any = Depends(get_app_state)):
    """FactCheckAgent 설정 조회"""
    try:
        factcheck_wrapper = state.registry.get("factcheck")
        
        request = {"action": "config"}
        result = await factcheck_wrapper.handle(request)
        
        if result["success"]:
            return JSONResponse(result["result"])
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except KeyError:
        raise HTTPException(status_code=503, detail="FactCheckAgent가 등록되지 않았습니다.")
    except Exception as e:
        logger.error(f"FactCheckAgent 설정 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/document/{doc_id}/cache/factcheck")
async def clear_factcheck_cache(doc_id: str, state: Any = Depends(get_app_state)):
    """특정 문서의 팩트체킹 분석 캐시 삭제"""
    try:
        factcheck_wrapper = state.registry.get("factcheck")
        
        request = {
            "action": "clear_cache",
            "doc_id": doc_id
        }
        result = await factcheck_wrapper.handle(request)
        
        if result["success"]:
            return JSONResponse(result["result"])
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except KeyError:
        raise HTTPException(status_code=503, detail="FactCheckAgent가 등록되지 않았습니다.")
    except Exception as e:
        logger.error(f"팩트체킹 캐시 삭제 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/factcheck/cache/search")
async def clear_factcheck_search_cache(state: Any = Depends(get_app_state)):
    """FactCheckAgent 검색 캐시 삭제"""
    try:
        factcheck_wrapper = state.registry.get("factcheck")
        
        request = {"action": "clear_search_cache"}
        result = await factcheck_wrapper.handle(request)
        
        if result["success"]:
            return JSONResponse(result["result"])
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except KeyError:
        raise HTTPException(status_code=503, detail="FactCheckAgent가 등록되지 않았습니다.")
    except Exception as e:
        logger.error(f"팩트체킹 검색 캐시 삭제 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ── 통합 분석 엔드포인트 ──

@app.post("/document/{doc_id}/analyze/full")
async def analyze_full(doc_id: str, state: Any = Depends(get_app_state)):
    """
    문서 전체 분석 (품질 + 팩트체크)
    """
    try:
        analyses = {}
        
        # 품질 분석
        try:
            quality_wrapper = state.registry.get("quality")
            quality_request = {"action": "analyze", "doc_id": doc_id}
            quality_result = await quality_wrapper.handle(quality_request)
            
            if quality_result["success"]:
                analyses["quality"] = quality_result["result"]
            else:
                analyses["quality"] = {"error": quality_result["error"]}
                
        except KeyError:
            analyses["quality"] = {"error": "QualityAgent가 등록되지 않았습니다"}
        
        # 팩트체크 분석
        try:
            factcheck_wrapper = state.registry.get("factcheck")
            factcheck_request = {"action": "analyze", "doc_id": doc_id}
            factcheck_result = await factcheck_wrapper.handle(factcheck_request)
            
            if factcheck_result["success"]:
                analyses["factcheck"] = factcheck_result["result"]
            else:
                analyses["factcheck"] = {"error": factcheck_result["error"]}
                
        except KeyError:
            analyses["factcheck"] = {"error": "FactCheckAgent가 등록되지 않았습니다"}
        
        # 통계 업데이트
        state.increment_stat("analyses_performed")
        
        # 전체 분석 요약
        total_issues = 0
        successful_analyses = []
        failed_analyses = []
        
        for analysis_type, result in analyses.items():
            if "error" in result:
                failed_analyses.append(analysis_type)
            else:
                successful_analyses.append(analysis_type)
                total_issues += result.get("total_issues", 0)
        
        return JSONResponse({
            "doc_id": doc_id,
            "analyses": analyses,
            "summary": {
                "total_issues": total_issues,
                "successful_analyses": successful_analyses,
                "failed_analyses": failed_analyses,
                "analysis_types": list(analyses.keys())
            },
            "message": f"전체 분석 완료: {len(successful_analyses)}개 분석 성공, {total_issues}개 이슈 발견"
        })
        
    except Exception as e:
        state.increment_stat("failed_requests")
        logger.error(f"전체 분석 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{doc_id}/info")
async def get_document_info(doc_id: str, state: Any = Depends(get_app_state)):
    """특정 문서의 메타데이터 조회"""
    try:
        document_wrapper = state.registry.get("document")
        
        request = {
            "action": "info",
            "doc_id": doc_id
        }
        result = await document_wrapper.handle(request)
        
        if result["success"]:
            return JSONResponse(result["result"])
        else:
            if "찾을 수 없습니다" in result["error"]:
                raise HTTPException(status_code=404, detail=result["error"])
            raise HTTPException(status_code=500, detail=result["error"])
            
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(status_code=503, detail="DocumentAgent가 등록되지 않았습니다.")
    except Exception as e:
        logger.error(f"문서 정보 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document/{doc_id}/stats")
async def get_document_stats(doc_id: str, state: Any = Depends(get_app_state)):
    """특정 문서의 통계 정보 조회"""
    try:
        document_wrapper = state.registry.get("document")
        
        request = {
            "action": "stats",
            "doc_id": doc_id
        }
        result = await document_wrapper.handle(request)
        
        if result["success"]:
            return JSONResponse(result["result"])
        else:
            if "찾을 수 없습니다" in result["error"]:
                raise HTTPException(status_code=404, detail=result["error"])
            raise HTTPException(status_code=500, detail=result["error"])
            
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(status_code=503, detail="DocumentAgent가 등록되지 않았습니다.")
    except Exception as e:
        logger.error(f"문서 통계 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document/{doc_id}/slides")
async def get_document_slides(
    doc_id: str, 
    page_id: Optional[str] = None,
    state: Any = Depends(get_app_state)
):
    """문서의 슬라이드 데이터 조회"""
    try:
        document_wrapper = state.registry.get("document")
        
        request = {
            "action": "slide",
            "doc_id": doc_id
        }
        
        if page_id:
            request["page_id"] = page_id
        
        result = await document_wrapper.handle(request)
        
        if result["success"]:
            return JSONResponse(result["result"])
        else:
            if "찾을 수 없습니다" in result["error"]:
                raise HTTPException(status_code=404, detail=result["error"])
            raise HTTPException(status_code=500, detail=result["error"])
            
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(status_code=503, detail="DocumentAgent가 등록되지 않았습니다.")
    except Exception as e:
        logger.error(f"슬라이드 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/document/{doc_id}/search")
async def search_document(
    doc_id: str,
    search_request: Dict[str, Any],
    state: Any = Depends(get_app_state)
):
    """
    문서 내 의미적 검색
    
    Body: {"query": "검색어", "top_k": 5}
    """
    try:
        query = search_request.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="검색어(query)가 필요합니다.")
        
        top_k = search_request.get("top_k", 5)
        
        document_wrapper = state.registry.get("document")
        
        request = {
            "action": "search",
            "doc_id": doc_id,
            "query": query,
            "top_k": top_k
        }
        result = await document_wrapper.handle(request)
        
        if result["success"]:
            return JSONResponse(result["result"])
        else:
            if "찾을 수 없습니다" in result["error"]:
                raise HTTPException(status_code=404, detail=result["error"])
            raise HTTPException(status_code=500, detail=result["error"])
            
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(status_code=503, detail="DocumentAgent가 등록되지 않았습니다.")
    except Exception as e:
        logger.error(f"문서 검색 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ── 에이전트 관리 엔드포인트들 ──

@app.get("/agents")
async def list_agents(state: Any = Depends(get_app_state)):
    """등록된 모든 에이전트 목록"""
    agents = {}
    for name, agent in state.registry.all().items():
        agents[name] = agent.get_info()
    
    return JSONResponse({
        "agents": agents,
        "total_agents": len(agents)
    })


@app.get("/agents/{agent_name}/info")
async def get_agent_info(agent_name: str, state: Any = Depends(get_app_state)):
    """특정 에이전트 정보"""
    try:
        agent = state.registry.get(agent_name)
        return JSONResponse(agent.get_info())
    except KeyError:
        raise HTTPException(status_code=404, detail=f"에이전트를 찾을 수 없습니다: {agent_name}")


# ── 개발용 엔드포인트들 ──

@app.post("/dev/reset-stats")
async def reset_stats(state: Any = Depends(get_app_state)):
    """통계 초기화 (개발용)"""
    state.reset_stats()
    
    # 모든 에이전트 통계도 초기화
    for agent in state.registry.all().values():
        agent.reset_stats()
    
    return {"message": "통계가 초기화되었습니다."}


@app.get("/dev/config")
async def get_config(state: Any = Depends(get_app_state)):
    """현재 설정 조회 (개발용)"""
    return JSONResponse(state.config)


@app.post("/dev/config")
async def update_config(
    config_updates: Dict[str, Any],
    state: Any = Depends(get_app_state)
):
    """설정 업데이트 (개발용)"""
    for key, value in config_updates.items():
        if key in state.config:
            state.update_config(key, value)
    
    return {"message": "설정이 업데이트되었습니다.", "config": state.config}


# ── 루트 엔드포인트 ──

@app.get("/")
async def root():
    """루트 엔드포인트 - API 정보"""
    return {
        "name": "EDU-Audit MCP Server",
        "version": "0.1.0",
        "description": "교육 자료 품질 검수 서비스",
        "endpoints": {
            "health": "/health",
            "upload": "POST /document/upload",
            "list": "/document/list",
            "info": "/document/{doc_id}/info",
            "search": "POST /document/{doc_id}/search",
            "quality_analyze": "POST /document/{doc_id}/analyze/quality",
            "quality_summary": "/document/{doc_id}/summary/quality",
            "factcheck_analyze": "POST /document/{doc_id}/analyze/factcheck",
            "factcheck_summary": "/document/{doc_id}/summary/factcheck",
            "full_analyze": "POST /document/{doc_id}/analyze/full",
            "cache_management": "DELETE /document/{doc_id}/cache/{agent_type}",
            "agents": "/agents"
        }
    }



# ── 서버 실행 ──

if __name__ == "__main__":
    import uvicorn
    
    # 개발 서버 실행
    uvicorn.run(
        "src.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )