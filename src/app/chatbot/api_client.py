# chatbot/api_client.py
"""
EDU-Audit API Client
"""
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
import aiohttp


class EduAuditClient:
    """FastAPI 서버와 통신하는 클라이언트"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def upload_document(self, file_path: str) -> Dict[str, Any]:
        """문서 업로드"""
        with open(file_path, "rb") as f:
            response = requests.post(
                f"{self.base_url}/document/upload",
                files={"file": f}
            )
        response.raise_for_status()
        return response.json()
    
    def list_documents(self) -> Dict[str, Any]:
        """문서 목록 조회"""
        response = requests.get(f"{self.base_url}/document/list")
        response.raise_for_status()
        return response.json()
    
    def get_document_info(self, doc_id: str) -> Dict[str, Any]:
        """특정 문서 정보 조회"""
        response = requests.get(f"{self.base_url}/document/{doc_id}/info")
        response.raise_for_status()
        return response.json()
    
    def analyze_quality(self, doc_id: str) -> Dict[str, Any]:
        """품질 분석"""
        response = requests.post(f"{self.base_url}/document/{doc_id}/analyze/quality", timeout=600)
        response.raise_for_status()
        return response.json()
    
    def analyze_factcheck(self, doc_id: str) -> Dict[str, Any]:
        """팩트체킹 분석"""
        response = requests.post(f"{self.base_url}/document/{doc_id}/analyze/factcheck", timeout=600)
        response.raise_for_status()
        return response.json()
    
    def analyze_full(self, doc_id: str) -> Dict[str, Any]:
        """전체 분석 (품질 + 팩트체킹)"""
        response = requests.post(f"{self.base_url}/document/{doc_id}/analyze/full", timeout=600)
        response.raise_for_status()
        return response.json()
    
    def search_document(self, doc_id: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        """문서 내 검색"""
        response = requests.post(
            f"{self.base_url}/document/{doc_id}/search",
            json={"query": query, "top_k": top_k}
        )
        response.raise_for_status()
        return response.json()