"""
EDU-Audit MCP Server
교육 콘텐츠 검수를 위한 MCP 도구 서버
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# MCP 관련 import
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# 우리 에이전트들
from src.agents.document_agent import MultimodalDocumentAgent
from src.agents.quality_agent import MultimodalQualityAgent
from src.agents.factcheck_agent import FactCheckAgent

# 모델들
from src.core.models import (
    DocumentMeta, PageInfo, Issue, AuditReport, 
    FactCheckRequest, FactCheckResult, QueryRequest, QueryResponse,
    generate_report_id
)

# 환경 설정
from dotenv import load_dotenv

load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EDUAuditMCPServer:
    """EDU-Audit MCP 서버"""
    
    def __init__(self):
        """MCP 서버 초기화"""
        self.server = Server("edu-audit")
        
        # API 키 설정
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.serpapi_key = os.getenv("SERPAPI_API_KEY")
        
        # 에이전트들 초기화
        self.document_agent = MultimodalDocumentAgent(
            openai_api_key=self.openai_api_key,
            enable_ocr=True,
            enable_vision_analysis=bool(self.openai_api_key)
        )
        
        self.quality_agent = MultimodalQualityAgent(
            openai_api_key=self.openai_api_key,
            enable_vision_analysis=bool(self.openai_api_key)
        )
        
        self.fact_agent = FactCheckAgent(
            openai_api_key=self.openai_api_key,
            serpapi_key=self.serpapi_key
        )
        
        # 현재 처리 중인 문서들 (메모리 캐시)
        self.active_documents: Dict[str, DocumentMeta] = {}
        self.audit_reports: Dict[str, AuditReport] = {}
        
        # MCP 도구들 등록
        self._register_tools()
        
        logger.info("EDU-Audit MCP 서버 초기화 완료")
    
    def _register_tools(self):
        """MCP 도구들 등록"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """사용 가능한 도구 목록 반환"""
            return [
                types.Tool(
                    name="run_full_audit",
                    description="교육 문서의 전체 검수를 수행합니다 (파싱 + 품질검사 + 사실검증)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "검수할 문서 파일 경로 (PDF 또는 PPT)"
                            },
                            "include_fact_check": {
                                "type": "boolean",
                                "description": "사실 검증 포함 여부 (기본값: true)",
                                "default": True
                            },
                            "quality_level": {
                                "type": "string",
                                "enum": ["basic", "standard", "comprehensive"],
                                "description": "품질 검사 수준 (기본값: standard)",
                                "default": "standard"
                            }
                        },
                        "required": ["file_path"]
                    }
                ),
                
                types.Tool(
                    name="query_content",
                    description="문서 내용에 대한 자연어 질의응답을 제공합니다",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "문서에 대한 질문"
                            },
                            "doc_id": {
                                "type": "string",
                                "description": "대상 문서 ID (선택사항, 없으면 최근 문서 사용)"
                            }
                        },
                        "required": ["question"]
                    }
                ),
                
                types.Tool(
                    name="check_typo",
                    description="특정 텍스트의 오탈자와 문법을 검사합니다",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "검사할 텍스트"
                            },
                            "language": {
                                "type": "string",
                                "enum": ["ko", "en", "auto"],
                                "description": "텍스트 언어 (기본값: auto)",
                                "default": "auto"
                            }
                        },
                        "required": ["text"]
                    }
                ),
                
                types.Tool(
                    name="fact_check",
                    description="특정 주장이나 문장의 사실 여부를 검증합니다",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sentence": {
                                "type": "string",
                                "description": "검증할 문장이나 주장"
                            },
                            "context": {
                                "type": "string",
                                "description": "추가 문맥 정보 (선택사항)"
                            }
                        },
                        "required": ["sentence"]
                    }
                ),
                
                types.Tool(
                    name="get_document_stats",
                    description="문서의 통계 정보를 조회합니다",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "문서 ID (선택사항, 없으면 최근 문서 사용)"
                            }
                        }
                    }
                ),
                
                types.Tool(
                    name="get_audit_report",
                    description="검수 보고서를 조회합니다",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "report_id": {
                                "type": "string",
                                "description": "보고서 ID (선택사항, 없으면 최근 보고서 사용)"
                            },
                            "format": {
                                "type": "string",
                                "enum": ["json", "summary", "detailed"],
                                "description": "보고서 형식 (기본값: summary)",
                                "default": "summary"
                            }
                        }
                    }
                ),
                
                types.Tool(
                    name="list_documents",
                    description="현재 로드된 문서 목록을 조회합니다",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            """도구 호출 처리"""
            try:
                if name == "run_full_audit":
                    return await self._handle_run_full_audit(arguments)
                elif name == "query_content":
                    return await self._handle_query_content(arguments)
                elif name == "check_typo":
                    return await self._handle_check_typo(arguments)
                elif name == "fact_check":
                    return await self._handle_fact_check(arguments)
                elif name == "get_document_stats":
                    return await self._handle_get_document_stats(arguments)
                elif name == "get_audit_report":
                    return await self._handle_get_audit_report(arguments)
                elif name == "list_documents":
                    return await self._handle_list_documents(arguments)
                else:
                    raise ValueError(f"알 수 없는 도구: {name}")
            
            except Exception as e:
                logger.error(f"도구 호출 오류 ({name}): {str(e)}")
                return [types.TextContent(
                    type="text",
                    text=f"오류 발생: {str(e)}"
                )]
    
    async def _handle_run_full_audit(self, arguments: dict) -> list[types.TextContent]:
        """전체 검수 실행"""
        file_path = arguments["file_path"]
        include_fact_check = arguments.get("include_fact_check", True)
        quality_level = arguments.get("quality_level", "standard")
        
        if not Path(file_path).exists():
            return [types.TextContent(
                type="text",
                text=f"파일을 찾을 수 없습니다: {file_path}"
            )]
        
        try:
            start_time = datetime.now()
            
            # 1단계: 문서 파싱
            logger.info(f"문서 파싱 시작: {file_path}")
            doc_meta = await self.document_agent.parse_document(file_path)
            pages = self.document_agent.get_pages(doc_meta.doc_id)
            
            # 문서 캐시에 저장
            self.active_documents[doc_meta.doc_id] = doc_meta
            
            # 2단계: 품질 검사
            logger.info(f"품질 검사 시작: {quality_level} 수준")
            quality_issues = await self.quality_agent.check_document(doc_meta, pages)
            
            all_issues = quality_issues.copy()
            
            # 3단계: 사실 검증 (옵션)
            if include_fact_check:
                logger.info("사실 검증 시작")
                fact_issues = await self.fact_agent.check_document_facts(doc_meta, pages)
                all_issues.extend(fact_issues)
            
            # 4단계: 보고서 생성
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            report = AuditReport(
                report_id=generate_report_id(doc_meta.doc_id),
                doc_id=doc_meta.doc_id,
                issues=all_issues,
                started_at=start_time,
                completed_at=end_time,
                processing_time=processing_time,
                agents_used=["document_agent", "quality_agent"] + (["fact_agent"] if include_fact_check else [])
            )
            
            # 보고서 캐시에 저장
            self.audit_reports[report.report_id] = report
            
            # 결과 요약 생성
            summary = self._generate_audit_summary(doc_meta, report, processing_time)
            
            return [types.TextContent(
                type="text",
                text=summary
            )]
        
        except Exception as e:
            logger.error(f"전체 검수 실패: {str(e)}")
            return [types.TextContent(
                type="text", 
                text=f"검수 실패: {str(e)}"
            )]
    
    async def _handle_query_content(self, arguments: dict) -> list[types.TextContent]:
        """문서 내용 질의응답"""
        question = arguments["question"]
        doc_id = arguments.get("doc_id")
        
        # 문서 선택
        if doc_id and doc_id in self.active_documents:
            target_doc_id = doc_id
        elif self.active_documents:
            # 가장 최근 문서 사용
            target_doc_id = list(self.active_documents.keys())[-1]
        else:
            return [types.TextContent(
                type="text",
                text="로드된 문서가 없습니다. 먼저 run_full_audit을 실행해주세요."
            )]
        
        try:
            # 문서 검색
            search_results = self.document_agent.search_in_document(
                target_doc_id, question, top_k=5
            )
            
            if not search_results:
                return [types.TextContent(
                    type="text", 
                    text=f"'{question}'에 관련된 내용을 찾을 수 없습니다."
                )]
            
            # 관련 이슈들 찾기
            relevant_issues = []
            if target_doc_id in [report.doc_id for report in self.audit_reports.values()]:
                for report in self.audit_reports.values():
                    if report.doc_id == target_doc_id:
                        relevant_issues = report.issues
                        break
            
            # 응답 생성
            response = QueryResponse(
                question=question,
                answer=self._generate_qa_answer(question, search_results, relevant_issues),
                relevant_issues=relevant_issues[:3],  # 상위 3개만
                confidence=0.8,
                generated_at=datetime.now()
            )
            
            return [types.TextContent(
                type="text",
                text=response.answer
            )]
        
        except Exception as e:
            logger.error(f"질의응답 실패: {str(e)}")
            return [types.TextContent(
                type="text",
                text=f"질의응답 실패: {str(e)}"
            )]
    
    async def _handle_check_typo(self, arguments: dict) -> list[types.TextContent]:
        """오탈자 검사"""
        text = arguments["text"]
        language = arguments.get("language", "auto")
        
        try:
            # 더미 문서 생성
            from src.core.models import generate_doc_id, PageInfo
            
            doc_meta = DocumentMeta(
                doc_id=generate_doc_id("typo_check"),
                title="오탈자 검사",
                doc_type="text",
                total_pages=1,
                file_path="temp"
            )
            
            page = PageInfo(
                page_id="p001",
                page_number=1,
                raw_text=text,
                word_count=len(text.split())
            )
            
            # 품질 검사 실행 (오탈자 위주)
            issues = await self.quality_agent.check_document(doc_meta, [page])
            
            if not issues:
                return [types.TextContent(
                    type="text",
                    text="오탈자가 발견되지 않았습니다."
                )]
            
            # 결과 포맷팅
            result_text = f"발견된 오탈자/문법 오류 ({len(issues)}개):\n\n"
            
            for i, issue in enumerate(issues, 1):
                result_text += f"{i}. '{issue.original_text}'\n"
                result_text += f"   문제: {issue.message}\n"
                result_text += f"   수정 제안: {issue.suggestion}\n"
                result_text += f"   신뢰도: {issue.confidence:.2f}\n\n"
            
            return [types.TextContent(
                type="text",
                text=result_text
            )]
        
        except Exception as e:
            logger.error(f"오탈자 검사 실패: {str(e)}")
            return [types.TextContent(
                type="text",
                text=f"오탈자 검사 실패: {str(e)}"
            )]
    
    async def _handle_fact_check(self, arguments: dict) -> list[types.TextContent]:
        """사실 검증"""
        sentence = arguments["sentence"]
        context = arguments.get("context")
        
        try:
            result = await self.fact_agent.verify_single_claim(sentence, context)
            
            result_text = f"사실 검증 결과:\n\n"
            result_text += f"검증 대상: {sentence}\n\n"
            result_text += f"판정: {'✅ 사실' if result.is_factual else '❌ 거짓/의심'}\n"
            result_text += f"신뢰도: {result.confidence:.2f}\n\n"
            result_text += f"근거:\n{result.explanation}\n"
            
            if result.sources:
                result_text += f"\n참고 출처 ({len(result.sources)}개):\n"
                for i, source in enumerate(result.sources, 1):
                    result_text += f"{i}. {source}\n"
            
            return [types.TextContent(
                type="text",
                text=result_text
            )]
        
        except Exception as e:
            logger.error(f"사실 검증 실패: {str(e)}")
            return [types.TextContent(
                type="text",
                text=f"사실 검증 실패: {str(e)}"
            )]
    
    async def _handle_get_document_stats(self, arguments: dict) -> list[types.TextContent]:
        """문서 통계 조회"""
        doc_id = arguments.get("doc_id")
        
        if doc_id and doc_id in self.active_documents:
            target_doc_id = doc_id
        elif self.active_documents:
            target_doc_id = list(self.active_documents.keys())[-1]
        else:
            return [types.TextContent(
                type="text",
                text="로드된 문서가 없습니다."
            )]
        
        try:
            stats = self.document_agent.get_multimodal_stats(target_doc_id)
            
            stats_text = f"문서 통계 정보:\n\n"
            stats_text += f"문서 ID: {stats['doc_id']}\n"
            stats_text += f"제목: {stats['title']}\n"
            stats_text += f"총 페이지: {stats['total_pages']}\n"
            stats_text += f"총 단어 수: {stats['total_words']:,}\n"
            stats_text += f"총 문자 수: {stats['total_chars']:,}\n"
            stats_text += f"페이지당 평균 단어: {stats['avg_words_per_page']:.1f}\n\n"
            
            stats_text += f"멀티모달 요소:\n"
            stats_text += f"  총 요소: {stats['elements']['total']}개\n"
            stats_text += f"  텍스트: {stats['elements']['text']}개\n"
            stats_text += f"  이미지: {stats['elements']['image']}개\n"
            stats_text += f"  표: {stats['elements']['table']}개\n"
            stats_text += f"  차트: {stats['elements']['chart']}개\n\n"
            
            stats_text += f"분석 기능:\n"
            stats_text += f"  OCR 추출: {stats['multimodal_features']['ocr_text_extracted']}개\n"
            stats_text += f"  AI 설명: {stats['multimodal_features']['ai_descriptions_generated']}개\n"
            stats_text += f"  인덱스 생성: {'✅' if stats['has_index'] else '❌'}\n"
            
            return [types.TextContent(
                type="text",
                text=stats_text
            )]
        
        except Exception as e:
            logger.error(f"문서 통계 조회 실패: {str(e)}")
            return [types.TextContent(
                type="text",
                text=f"문서 통계 조회 실패: {str(e)}"
            )]
    
    async def _handle_get_audit_report(self, arguments: dict) -> list[types.TextContent]:
        """검수 보고서 조회"""
        report_id = arguments.get("report_id")
        format_type = arguments.get("format", "summary")
        
        if report_id and report_id in self.audit_reports:
            target_report = self.audit_reports[report_id]
        elif self.audit_reports:
            target_report = list(self.audit_reports.values())[-1]
        else:
            return [types.TextContent(
                type="text",
                text="생성된 검수 보고서가 없습니다."
            )]
        
        try:
            if format_type == "json":
                report_json = target_report.model_dump_json(indent=2)
                return [types.TextContent(
                    type="text",
                    text=f"검수 보고서 (JSON):\n\n```json\n{report_json}\n```"
                )]
            
            elif format_type == "detailed":
                report_text = self._generate_detailed_report(target_report)
            else:  # summary
                report_text = self._generate_summary_report(target_report)
            
            return [types.TextContent(
                type="text",
                text=report_text
            )]
        
        except Exception as e:
            logger.error(f"보고서 조회 실패: {str(e)}")
            return [types.TextContent(
                type="text",
                text=f"보고서 조회 실패: {str(e)}"
            )]
    
    async def _handle_list_documents(self, arguments: dict) -> list[types.TextContent]:
        """문서 목록 조회"""
        if not self.active_documents:
            return [types.TextContent(
                type="text",
                text="로드된 문서가 없습니다."
            )]
        
        docs_text = f"로드된 문서 목록 ({len(self.active_documents)}개):\n\n"
        
        for i, (doc_id, doc_meta) in enumerate(self.active_documents.items(), 1):
            docs_text += f"{i}. {doc_meta.title}\n"
            docs_text += f"   ID: {doc_id}\n"
            docs_text += f"   타입: {doc_meta.doc_type}\n"
            docs_text += f"   페이지: {doc_meta.total_pages}\n"
            docs_text += f"   파일: {doc_meta.file_path}\n"
            docs_text += f"   생성일: {doc_meta.created_at.strftime('%Y-%m-%d %H:%M')}\n\n"
        
        return [types.TextContent(
            type="text",
            text=docs_text
        )]
    
    def _generate_audit_summary(self, doc_meta: DocumentMeta, report: AuditReport, processing_time: float) -> str:
        """검수 요약 생성"""
        summary = f"📋 교육 콘텐츠 검수 완료\n\n"
        summary += f"문서: {doc_meta.title}\n"
        summary += f"파일: {doc_meta.file_path}\n"
        summary += f"페이지: {doc_meta.total_pages}\n"
        summary += f"처리 시간: {processing_time:.1f}초\n\n"
        
        summary += f"📊 검수 결과:\n"
        summary += f"  총 이슈: {report.total_issues}개\n"
        summary += f"  텍스트 이슈: {report.text_issues}개\n"
        summary += f"  이미지 이슈: {report.image_issues}개\n"
        summary += f"  표 이슈: {report.table_issues}개\n"
        summary += f"  차트 이슈: {report.chart_issues}개\n\n"
        
        if report.issues:
            summary += f"🔍 주요 이슈 (상위 3개):\n"
            for i, issue in enumerate(report.issues[:3], 1):
                summary += f"  {i}. [{issue.issue_type.value.upper()}] {issue.message[:60]}...\n"
                summary += f"     신뢰도: {issue.confidence:.2f} | 페이지: {issue.page_id}\n"
        
        summary += f"\n보고서 ID: {report.report_id}\n"
        summary += f"상세 보고서는 get_audit_report 도구로 조회 가능합니다."
        
        return summary
    
    def _generate_qa_answer(self, question: str, search_results: List[Dict], issues: List[Issue]) -> str:
        """질의응답 답변 생성"""
        answer = f"질문: {question}\n\n"
        
        if search_results:
            answer += f"📄 관련 내용:\n"
            for i, result in enumerate(search_results[:3], 1):
                answer += f"{i}. {result['text'][:100]}...\n"
                answer += f"   (페이지: {result.get('page_id', 'N/A')}, 신뢰도: {result.get('score', 0):.2f})\n\n"
        
        # 관련 이슈가 있다면 언급
        relevant_issues = [issue for issue in issues if question.lower() in issue.message.lower() or question.lower() in issue.original_text.lower()]
        
        if relevant_issues:
            answer += f"⚠️ 관련 품질 이슈:\n"
            for issue in relevant_issues[:2]:
                answer += f"- {issue.message}\n"
        
        return answer
    
    def _generate_summary_report(self, report: AuditReport) -> str:
        """요약 보고서 생성"""
        doc_meta = self.active_documents.get(report.doc_id)
        
        text = f"📋 검수 보고서 요약\n\n"
        text += f"보고서 ID: {report.report_id}\n"
        text += f"문서: {doc_meta.title if doc_meta else 'Unknown'}\n"
        text += f"생성일: {report.completed_at.strftime('%Y-%m-%d %H:%M:%S') if report.completed_at else 'N/A'}\n"
        text += f"처리 시간: {report.processing_time:.1f}초\n\n"
        
        text += f"📊 이슈 통계:\n"
        text += f"  총 이슈: {report.total_issues}개\n"
        text += f"  텍스트: {report.text_issues}개\n"
        text += f"  이미지: {report.image_issues}개\n"
        text += f"  표: {report.table_issues}개\n"
        text += f"  차트: {report.chart_issues}개\n\n"
        
        if report.issues:
            # 신뢰도별 분류
            high_conf = [i for i in report.issues if i.confidence >= 0.8]
            medium_conf = [i for i in report.issues if 0.5 <= i.confidence < 0.8]
            low_conf = [i for i in report.issues if i.confidence < 0.5]
            
            text += f"🎯 신뢰도별 분포:\n"
            text += f"  높음 (≥0.8): {len(high_conf)}개\n"
            text += f"  중간 (0.5-0.8): {len(medium_conf)}개\n"
            text += f"  낮음 (<0.5): {len(low_conf)}개\n"
        
        return text
    
    def _generate_detailed_report(self, report: AuditReport) -> str:
        """상세 보고서 생성"""
        text = self._generate_summary_report(report)
        
        if report.issues:
            text += f"\n🔍 발견된 이슈 상세:\n\n"
            
            for i, issue in enumerate(report.issues, 1):
                text += f"{i}. [{issue.issue_type.value.upper()}] {issue.original_text[:30]}...\n"
                text += f"   메시지: {issue.message}\n"
                text += f"   제안: {issue.suggestion}\n"
                text += f"   신뢰도: {issue.confidence:.2f}\n"
                text += f"   위치: {issue.page_id}"
                if issue.element_id:
                    text += f" (요소: {issue.element_id})"
                text += f"\n   검출자: {issue.agent_name}\n\n"
        
        return text


async def main():
    """MCP 서버 실행"""
    server_instance = EDUAuditMCPServer()
    
    # stdio를 통한 MCP 서버 실행
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="edu-audit",
                server_version="1.0.0",
                capabilities=server_instance.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())