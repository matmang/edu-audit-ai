# chatbot/tools.py
"""
LangChain Tools for EDU-Audit
"""
from langchain.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
from api_client import EduAuditClient
import json


# ── Tool Input Schemas ──

class DocumentListInput(BaseModel):
    """문서 목록 조회 입력 (인자 없음)"""
    dummy: Optional[str] = Field(default="", description="사용하지 않는 더미 파라미터")


class DocumentInfoInput(BaseModel):
    """문서 정보 조회 입력"""
    doc_id: str = Field(..., description="조회할 문서 ID")


class AnalysisInput(BaseModel):
    """분석 입력"""
    doc_id: str = Field(..., description="분석할 문서 ID")


class SearchInput(BaseModel):
    """검색 입력"""
    doc_id: str = Field(..., description="검색할 문서 ID")
    query: str = Field(..., description="검색 키워드")
    top_k: int = Field(default=5, description="반환할 결과 수")


# ── LangChain Tools ──

class ListDocumentsTool(BaseTool):
    """업로드된 문서 목록 조회 도구"""
    name: str = "list_documents"
    description: str = """
    업로드된 모든 문서의 목록을 조회합니다.
    사용자가 "문서 목록 보여줘", "어떤 문서들이 있어?" 같은 질문을 할 때 사용하세요.
    입력값은 필요 없습니다 (dummy 파라미터를 빈 문자열로 전달).
    """
    args_schema: Type[BaseModel] = DocumentListInput
    client: EduAuditClient = None
    
    def __init__(self, client: EduAuditClient):
        super().__init__()
        self.client = client
    
    def _run(self, dummy: str = "") -> str:
        """동기 실행"""
        try:
            result = self.client.list_documents()
            print(result)
            docs = result.get("documents", [])
            
            if not docs:
                return "업로드된 문서가 없습니다."
            
            summary = f"총 {len(docs)}개의 문서가 있습니다:\n\n"
            for doc in docs:
                summary += f"- ID: {doc['doc_id']}\n"
                summary += f"  문서 설명: {doc['title']}\n"
                summary += f"  페이지 수: {doc['total_pages']}\n"
                summary += f"  업로드: {doc['created_at']}\n\n"
            
            return summary
        except Exception as e:
            return f"문서 목록 조회 중 오류 발생: {str(e)}"
    
    async def _arun(self, dummy: str = "") -> str:
        """비동기 실행"""
        return self._run(dummy)


class GetDocumentInfoTool(BaseTool):
    """특정 문서의 상세 정보 조회 도구"""
    name: str = "get_document_info"
    description: str = """
    특정 문서의 상세 정보를 조회합니다.
    doc_id를 입력으로 받습니다.
    사용자가 특정 문서에 대해 물어볼 때 사용하세요.
    """
    args_schema: Type[BaseModel] = DocumentInfoInput
    client: EduAuditClient = None
    
    def __init__(self, client: EduAuditClient):
        super().__init__()
        self.client = client
    
    def _run(self, doc_id: str) -> str:
        try:
            result = self.client.get_document_info(doc_id)
            doc_meta = result.get("doc_meta")
            
            info = f"문서 정보:\n"
            info += f"- ID: {result['doc_id']}\n"
            info += f"- 문서 제목: {doc_meta['title']}\n"
            info += f"- 페이지 수: {doc_meta['total_pages']}\n"
            info += f"- 생성일: {doc_meta['created_at']}\n"
            
            return info
        except Exception as e:
            return f"문서 정보 조회 실패: {str(e)}"
    
    async def _arun(self, doc_id: str) -> str:
        return self._run(doc_id)


class AnalyzeQualityTool(BaseTool):
    """문서 품질 분석 도구"""
    name: str = "analyze_quality"
    description: str = """
    문서의 품질을 분석합니다 (오탈자, 문법, 가독성 등).
    doc_id를 입력으로 받습니다.
    사용자가 "품질 검사해줘", "오탈자 찾아줘" 같은 요청을 할 때 사용하세요.
    """
    args_schema: Type[BaseModel] = AnalysisInput
    client: EduAuditClient = None
    
    def __init__(self, client: EduAuditClient):
        super().__init__()
        self.client = client
    
    def _run(self, doc_id: str) -> str:
        try:
            result = self.client.analyze_quality(doc_id)

            print(result)
            
            summary = f"품질 분석 결과:\n"
            summary += f"- 총 이슈: {result['total_issues']}개\n"
            
            if result['total_issues'] > 0:
                summary += "발견된 문제:\n"
                for issue in result['issues'][:5]:  # 상위 5개만
                    summary += f"  • [{issue['confidence_level']}] page: {issue['page_id']} {issue['issue_type']}: {issue['message']}\n"
                
                if result['total_issues'] > 5:
                    summary += f"  ... 외 {result['total_issues'] - 5}개\n"
            else:
                summary += "문제가 발견되지 않았습니다! ✅\n"
            
            return summary
        except Exception as e:
            return f"품질 분석 실패: {str(e)}"
    
    async def _arun(self, doc_id: str) -> str:
        return self._run(doc_id)


class AnalyzeFactCheckTool(BaseTool):
    """팩트체킹 분석 도구"""
    name: str = "analyze_factcheck"
    description: str = """
    문서 내용의 사실 여부를 검증합니다 (외부 검색 활용).
    doc_id를 입력으로 받습니다.
    사용자가 "사실 확인해줘", "팩트체크해줘" 같은 요청을 할 때 사용하세요.
    """
    args_schema: Type[BaseModel] = AnalysisInput
    client: EduAuditClient = None
    
    def __init__(self, client: EduAuditClient):
        super().__init__()
        self.client = client
    
    def _run(self, doc_id: str) -> str:
        try:
            result = self.client.analyze_factcheck(doc_id)
            print(result)
            
            summary = f"팩트체킹 결과:\n"
            summary += f"- 총 이슈: {result['total_issues']}개\n"
            
            if result['total_issues'] > 0:
                summary += "발견된 문제:\n"
                for issue in result['issues'][:5]:
                    summary += f"  • [{issue['confidence_level']}] {issue['message']}\n"
                
                if result['total_issues'] > 5:
                    summary += f"  ... 외 {result['total_issues'] - 5}개\n"
            else:
                summary += "사실 관계 오류가 발견되지 않았습니다! ✅\n"
            
            return summary
        except Exception as e:
            return f"팩트체킹 실패: {str(e)}"
    
    async def _arun(self, doc_id: str) -> str:
        return self._run(doc_id)


class AnalyzeFullTool(BaseTool):
    """전체 분석 도구 (품질 + 팩트체킹)"""
    name: str = "analyze_full"
    description: str = """
    문서의 품질과 팩트체킹을 동시에 수행합니다.
    doc_id를 입력으로 받습니다.
    사용자가 "전체 검수해줘", "완전히 분석해줘" 같은 요청을 할 때 사용하세요.
    """
    args_schema: Type[BaseModel] = AnalysisInput
    client: EduAuditClient = None
    
    def __init__(self, client: EduAuditClient):
        super().__init__()
        self.client = client
    
    def _run(self, doc_id: str) -> str:
        try:
            result = self.client.analyze_full(doc_id)
            print(result)
            
            summary = f"전체 분석 완료!\n\n"
            summary += f"📊 요약:\n"
            summary += f"- 총 이슈: {result['summary']['total_issues']}개\n"
            summary += f"- 성공한 분석: {', '.join(result['summary']['successful_analyses'])}\n"
            
            if result['summary']['failed_analyses']:
                summary += f"- 실패한 분석: {', '.join(result['summary']['failed_analyses'])}\n"
            
            summary += f"\n{result['message']}\n"
            
            return summary
        except Exception as e:
            return f"전체 분석 실패: {str(e)}"
    
    async def _arun(self, doc_id: str) -> str:
        return self._run(doc_id)


class SearchDocumentTool(BaseTool):
    """문서 내 검색 도구"""
    name: str = "search_document"
    description: str = """
    문서 내에서 특정 내용을 검색합니다 (의미 기반 검색).
    doc_id와 검색 키워드(query)를 입력으로 받습니다.
    사용자가 "문서에서 ~~ 찾아줘" 같은 요청을 할 때 사용하세요.
    """
    args_schema: Type[BaseModel] = SearchInput
    client: EduAuditClient = None
    
    def __init__(self, client: EduAuditClient):
        super().__init__()
        self.client = client
    
    def _run(self, doc_id: str, query: str, top_k: int = 5) -> str:
        try:
            result = self.client.search_document(doc_id, query, top_k)
            print(result)
            
            if not result.get('results'):
                return f"'{query}'에 대한 검색 결과가 없습니다."
            
            summary = f"'{query}' 검색 결과 (상위 {len(result['results'])}개):\n\n"
            
            for idx, item in enumerate(result['results'], 1):
                summary += f"{idx}. [페이지 {item['page_number']}] (유사도: {item['score']:.2f})\n"
                summary += f"   {item['text'][:150]}...\n\n"
            
            return summary
        except Exception as e:
            return f"검색 실패: {str(e)}"
    
    async def _arun(self, doc_id: str, query: str, top_k: int = 5) -> str:
        return self._run(doc_id, query, top_k)


# ── Tool Factory ──

def create_edu_audit_tools(base_url: str = "http://localhost:8000") -> list:
    """EDU-Audit Tools 생성"""
    client = EduAuditClient(base_url)
    
    return [
        ListDocumentsTool(client),
        GetDocumentInfoTool(client),
        AnalyzeQualityTool(client),
        AnalyzeFactCheckTool(client),
        AnalyzeFullTool(client),
        SearchDocumentTool(client),
    ]