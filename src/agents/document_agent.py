"""
EDU-Audit Multimodal Document Agent
PDF/PPT 멀티모달 파싱 및 LlamaIndex 연동 - 하이브리드 접근
"""

import asyncio
import logging
import os
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO

# PDF/PPT 파싱 라이브러리
import pdfplumber
from pptx import Presentation
from PIL import Image
import fitz  # PyMuPDF for image extraction

# OCR 라이브러리
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("EasyOCR not available. Install with: pip install easyocr")

# LlamaIndex 관련
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# 확장된 모델들
from src.core.models import (
    DocumentMeta, PageInfo, PageElement, ElementType,
    ImageElement, TableElement, ChartElement, BoundingBox,
    generate_doc_id, generate_element_id
)
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[2] / '.env.dev'
load_dotenv(env_path)

logger = logging.getLogger(__name__)

class MultimodalDocumentAgent:
    """멀티모달 문서 파싱 및 LlamaIndex 관리 에이전트"""
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        vision_model: str = "gpt-4-vision-preview",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        enable_ocr: bool = True,
        enable_vision_analysis: bool = True
    ):
        self.openai_api_key = openai_api_key
        self.vision_model = vision_model
        self.enable_ocr = enable_ocr and OCR_AVAILABLE
        self.enable_vision_analysis = enable_vision_analysis and openai_api_key
        
        # LlamaIndex 컴포넌트
        self.embeddings = OpenAIEmbedding(api_key=openai_api_key)
        self.text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Vision LLM (이미지 분석용)
        self.vision_llm = None
        if self.enable_vision_analysis:
            self.vision_llm = OpenAI(
                model=vision_model,
                api_key=openai_api_key,
                temperature=0.1
            )
        
        # OCR 초기화
        self.ocr_reader = None
        if self.enable_ocr:
            try:
                self.ocr_reader = easyocr.Reader(['ko', 'en'])
                logger.info("EasyOCR 초기화 완료 (한국어, 영어)")
            except Exception as e:
                logger.warning(f"OCR 초기화 실패: {str(e)}")
                self.enable_ocr = False
        
        # 메모리에 저장된 문서들
        self.documents: Dict[str, DocumentMeta] = {}
        self.pages: Dict[str, List[PageInfo]] = {}  # doc_id -> pages
        self.indexes: Dict[str, VectorStoreIndex] = {}  # doc_id -> index
        
        logger.info(f"MultimodalDocumentAgent 초기화 완료")
        logger.info(f"  OCR: {'활성화' if self.enable_ocr else '비활성화'}")
        logger.info(f"  Vision 분석: {'활성화' if self.enable_vision_analysis else '비활성화'}")
    
    async def parse_document(self, file_path: str) -> DocumentMeta:
        """
        멀티모달 문서 파싱 및 LlamaIndex 인덱싱
        
        Args:
            file_path: 파싱할 문서 파일 경로
            
        Returns:
            DocumentMeta: 파싱된 문서 메타데이터
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        logger.info(f"멀티모달 문서 파싱 시작: {file_path}")
        
        # 문서 ID 생성
        doc_id = generate_doc_id(str(file_path))
        
        # 파일 확장자에 따른 파싱
        if file_path.suffix.lower() == '.pdf':
            doc_meta, pages = await self._parse_pdf_multimodal(file_path, doc_id)
        elif file_path.suffix.lower() in ['.ppt', '.pptx']:
            doc_meta, pages = await self._parse_ppt_multimodal(file_path, doc_id)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")
        
        # 멀티모달 통계 업데이트
        doc_meta.total_images = sum(page.image_count for page in pages)
        doc_meta.total_tables = sum(page.table_count for page in pages)
        doc_meta.total_charts = sum(page.chart_count for page in pages)
        
        # 메모리에 저장
        self.documents[doc_id] = doc_meta
        self.pages[doc_id] = pages
        
        # 멀티모달 LlamaIndex 인덱스 생성
        await self._create_multimodal_index(doc_id, pages)
        
        logger.info(f"멀티모달 문서 파싱 완료: {doc_id}")
        logger.info(f"  총 {len(pages)} 페이지, {doc_meta.total_images} 이미지, {doc_meta.total_tables} 표, {doc_meta.total_charts} 차트")
        return doc_meta
    
    async def _parse_pdf_multimodal(self, file_path: Path, doc_id: str) -> Tuple[DocumentMeta, List[PageInfo]]:
        """PDF 멀티모달 파싱"""
        logger.info(f"PDF 멀티모달 파싱 중: {file_path}")
        
        pages = []
        title = None
        
        try:
            # PyMuPDF로 이미지 추출용
            pdf_doc = fitz.open(file_path)
            
            with pdfplumber.open(file_path) as pdf:
                # 첫 페이지에서 제목 추출
                if pdf.pages:
                    first_page_text = pdf.pages[0].extract_text() or ""
                    title = self._extract_title_from_text(first_page_text)
                
                # 각 페이지 파싱
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"페이지 {page_num} 파싱 중...")
                    
                    # 텍스트 추출
                    page_text = page.extract_text() or ""
                    
                    # 멀티모달 요소들 추출
                    elements = []
                    
                    # 1. 텍스트 요소 추가
                    if page_text.strip():
                        text_element = PageElement(
                            element_id=generate_element_id(f"p{page_num:03d}", ElementType.TEXT, 0),
                            element_type=ElementType.TEXT,
                            text_content=page_text,
                            confidence=1.0
                        )
                        elements.append(text_element)
                    
                    # 2. 이미지 추출 및 분석
                    fitz_page = pdf_doc[page_num - 1]
                    image_elements = await self._extract_images_from_page(
                        page, fitz_page, f"p{page_num:03d}"
                    )
                    elements.extend(image_elements)
                    
                    # 3. 표 추출
                    table_elements = await self._extract_tables_from_page(
                        page, f"p{page_num:03d}"
                    )
                    elements.extend(table_elements)
                    
                    # 4. 차트 감지 (이미지 기반)
                    chart_elements = await self._detect_charts_in_images(
                        image_elements, f"p{page_num:03d}"
                    )
                    elements.extend(chart_elements)
                    
                    # PageInfo 생성
                    page_info = PageInfo(
                        page_id=f"p{page_num:03d}",
                        page_number=page_num,
                        raw_text=page_text,
                        word_count=len(page_text.split()) if page_text else 0,
                        elements=elements
                    )
                    
                    pages.append(page_info)
                    
                    # 진행상황 로그
                    if page_num % 5 == 0:
                        logger.info(f"PDF 파싱 진행: {page_num}/{len(pdf.pages)} 페이지")
        
        except Exception as e:
            logger.error(f"PDF 파싱 실패: {str(e)}")
            raise
        finally:
            if 'pdf_doc' in locals():
                pdf_doc.close()
        
        # 문서 메타데이터 생성
        doc_meta = DocumentMeta(
            doc_id=doc_id,
            title=title or file_path.stem,
            doc_type="pdf",
            total_pages=len(pages),
            file_path=str(file_path)
        )
        
        return doc_meta, pages
    
    async def _parse_ppt_multimodal(self, file_path: Path, doc_id: str) -> Tuple[DocumentMeta, List[PageInfo]]:
        """PowerPoint 멀티모달 파싱"""
        logger.info(f"PPT 멀티모달 파싱 중: {file_path}")
        
        pages = []
        title = None
        
        try:
            prs = Presentation(file_path)
            
            # 첫 슬라이드에서 제목 추출
            if prs.slides:
                first_slide_text = self._extract_slide_text(prs.slides[0])
                title = self._extract_title_from_text(first_slide_text)
            
            # 각 슬라이드 파싱
            for slide_num, slide in enumerate(prs.slides, 1):
                logger.info(f"슬라이드 {slide_num} 파싱 중...")
                
                elements = []
                slide_text_parts = []
                
                # 슬라이드 내 모든 shape 분석
                element_index = 0
                
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        # 텍스트 요소
                        text_element = PageElement(
                            element_id=generate_element_id(f"slide_{slide_num:03d}", ElementType.TEXT, element_index),
                            element_type=ElementType.TEXT,
                            bbox=self._shape_to_bbox(shape),
                            text_content=shape.text.strip(),
                            confidence=1.0
                        )
                        elements.append(text_element)
                        slide_text_parts.append(shape.text.strip())
                        element_index += 1
                    
                    elif shape.shape_type == 13: # msoPicture
                        # 이미지 요소
                        image_element = await self._extract_ppt_image(
                            shape, f"slide_{slide_num:03d}", element_index
                        )
                        if image_element:
                            elements.append(image_element)
                            element_index += 1
                    
                    elif shape.has_table:
                        # 표 요소
                        table_element = await self._extract_ppt_table(
                            shape.table, f"slide_{slide_num:03d}", element_index
                        )
                        if table_element:
                            elements.append(table_element)
                            element_index += 1
                
                # 전체 슬라이드 텍스트
                slide_text = "\n".join(slide_text_parts)
                
                # PageInfo 생성
                page_info = PageInfo(
                    page_id=f"slide_{slide_num:03d}",
                    page_number=slide_num,
                    raw_text=slide_text,
                    word_count=len(slide_text.split()) if slide_text else 0,
                    elements=elements
                )
                
                pages.append(page_info)
        
        except Exception as e:
            logger.error(f"PPT 파싱 실패: {str(e)}")
            raise
        
        # 문서 메타데이터 생성
        doc_meta = DocumentMeta(
            doc_id=doc_id,
            title=title or file_path.stem,
            doc_type="ppt",
            total_pages=len(pages),
            file_path=str(file_path)
        )
        
        return doc_meta, pages
    
    async def _extract_images_from_page(self, pdfplumber_page, fitz_page, page_id: str) -> List[PageElement]:
        """PDF 페이지에서 이미지 추출 및 분석"""
        elements = []
        
        try:
            # PyMuPDF로 이미지 추출
            image_list = fitz_page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # 이미지 데이터 추출
                    xref = img[0]
                    pix = fitz.Pixmap(fitz_page.parent, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.pil_tobytes(format="PNG")
                        pil_image = Image.open(BytesIO(img_data))
                        
                        # 이미지를 base64로 인코딩
                        buffered = BytesIO()
                        pil_image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        
                        # 이미지 분석
                        image_analysis = await self._analyze_image(pil_image, img_base64)
                        
                        # 이미지 요소 생성
                        image_element = PageElement(
                            element_id=generate_element_id(page_id, ElementType.IMAGE, img_index),
                            element_type=ElementType.IMAGE,
                            bbox=BoundingBox(x=0, y=0, width=pil_image.width, height=pil_image.height),
                            image_data=ImageElement(
                                element_id=generate_element_id(page_id, ElementType.IMAGE, img_index),
                                bbox=BoundingBox(x=0, y=0, width=pil_image.width, height=pil_image.height),
                                format="png",
                                size_bytes=len(img_data),
                                dimensions=(pil_image.width, pil_image.height),
                                ocr_text=image_analysis.get("ocr_text"),
                                description=image_analysis.get("description"),
                                image_data=f"data:image/png;base64,{img_base64}"
                            ),
                            confidence=image_analysis.get("confidence", 0.8)
                        )
                        
                        elements.append(image_element)
                        
                    pix = None  # 메모리 해제
                    
                except Exception as e:
                    logger.warning(f"이미지 {img_index} 처리 실패: {str(e)}")
        
        except Exception as e:
            logger.warning(f"페이지 이미지 추출 실패: {str(e)}")
        
        return elements
    
    async def _analyze_image(self, pil_image: Image.Image, img_base64: str) -> Dict[str, Any]:
        """이미지 분석 (OCR + Vision LLM)"""
        analysis = {
            "ocr_text": None,
            "description": None,
            "confidence": 0.5
        }
        
        # 1. OCR 분석
        if self.enable_ocr and self.ocr_reader:
            try:
                # PIL Image를 numpy array로 변환
                import numpy as np
                img_array = np.array(pil_image)
                
                ocr_results = self.ocr_reader.readtext(img_array)
                if ocr_results:
                    ocr_texts = [result[1] for result in ocr_results if result[2] > 0.5]
                    analysis["ocr_text"] = " ".join(ocr_texts)
                    analysis["confidence"] = max(analysis["confidence"], 0.7)
                
            except Exception as e:
                logger.warning(f"OCR 분석 실패: {str(e)}")
        
        # 2. Vision LLM 분석
        if self.enable_vision_analysis and self.vision_llm:
            try:
                description = await self._get_image_description(img_base64)
                if description:
                    analysis["description"] = description
                    analysis["confidence"] = max(analysis["confidence"], 0.8)
                
            except Exception as e:
                logger.warning(f"Vision LLM 분석 실패: {str(e)}")
        
        return analysis
    
    async def _get_image_description(self, img_base64: str) -> Optional[str]:
        """Vision LLM으로 이미지 설명 생성"""
        if not self.vision_llm:
            return None
        
        try:
            prompt = """이 이미지를 교육 콘텐츠 검수 관점에서 간단히 설명해주세요.
            
다음 정보를 포함해주세요:
- 이미지 타입 (다이어그램, 차트, 사진, 스크린샷 등)
- 주요 내용 (한 줄 요약)
- 텍스트가 있다면 주요 내용
- 교육적 목적 (설명, 예시, 데이터 등)

간단하고 명확하게 작성해주세요."""
            
            # OpenAI Vision API는 별도 구현 필요
            # 현재는 플레이스홀더
            response = await self._call_vision_api(prompt, img_base64)
            return response
            
        except Exception as e:
            logger.warning(f"이미지 설명 생성 실패: {str(e)}")
            return None
    
    async def _call_vision_api(self, prompt: str, img_base64: str) -> Optional[str]:
        """Vision API 호출 (OpenAI GPT-4V)"""
        # 실제 구현에서는 OpenAI의 vision API를 호출
        # 현재는 더미 응답
        await asyncio.sleep(0.1)  # API 호출 시뮬레이션
        return "다이어그램: 신경망 구조를 보여주는 교육용 이미지"
    
    async def _extract_tables_from_page(self, page, page_id: str) -> List[PageElement]:
        """PDF 페이지에서 표 추출"""
        elements = []
        
        try:
            tables = page.extract_tables()
            
            for table_index, table_data in enumerate(tables):
                if not table_data or len(table_data) < 2:
                    continue
                
                # 헤더와 데이터 분리
                headers = table_data[0] if table_data else []
                rows = table_data[1:] if len(table_data) > 1 else []
                
                # None 값 정리
                headers = [str(cell) if cell is not None else "" for cell in headers]
                clean_rows = []
                for row in rows:
                    clean_row = [str(cell) if cell is not None else "" for cell in row]
                    clean_rows.append(clean_row)
                
                table_element = PageElement(
                    element_id=generate_element_id(page_id, ElementType.TABLE, table_index),
                    element_type=ElementType.TABLE,
                    table_data=TableElement(
                        element_id=generate_element_id(page_id, ElementType.TABLE, table_index),
                        headers=headers,
                        rows=clean_rows,
                        extraction_confidence=0.9
                    ),
                    confidence=0.9
                )
                
                elements.append(table_element)
        
        except Exception as e:
            logger.warning(f"표 추출 실패: {str(e)}")
        
        return elements
    
    async def _detect_charts_in_images(self, image_elements: List[PageElement], page_id: str) -> List[PageElement]:
        """이미지에서 차트 감지"""
        chart_elements = []
        
        for img_element in image_elements:
            if not img_element.image_data or not img_element.image_data.description:
                continue
            
            description = img_element.image_data.description.lower()
            
            # 간단한 키워드 기반 차트 감지
            chart_keywords = ["차트", "그래프", "도표", "chart", "graph", "plot"]
            
            if any(keyword in description for keyword in chart_keywords):
                # 차트 요소로 변환
                chart_element = PageElement(
                    element_id=img_element.element_id.replace("image", "chart"),
                    element_type=ElementType.CHART,
                    bbox=img_element.bbox,
                    chart_data=ChartElement(
                        element_id=img_element.element_id.replace("image", "chart"),
                        bbox=img_element.bbox,
                        chart_type="unknown",
                        description=img_element.image_data.description
                    ),
                    confidence=0.7
                )
                
                chart_elements.append(chart_element)
        
        return chart_elements
    
    async def _extract_ppt_image(self, shape, page_id: str, element_index: int) -> Optional[PageElement]:
        """PPT에서 이미지 추출"""
        try:
            # PPT 이미지 처리는 복잡하므로 기본 구현만
            image_element = PageElement(
                element_id=generate_element_id(page_id, ElementType.IMAGE, element_index),
                element_type=ElementType.IMAGE,
                bbox=self._shape_to_bbox(shape),
                image_data=ImageElement(
                    element_id=generate_element_id(page_id, ElementType.IMAGE, element_index),
                    bbox=self._shape_to_bbox(shape),
                    format="unknown",
                    size_bytes=0,
                    dimensions=(100, 100),
                    description="PPT 이미지"
                ),
                confidence=0.6
            )
            
            return image_element
            
        except Exception as e:
            logger.warning(f"PPT 이미지 추출 실패: {str(e)}")
            return None
    
    async def _extract_ppt_table(self, table, page_id: str, element_index: int) -> Optional[PageElement]:
        """PPT에서 표 추출"""
        try:
            headers, rows = [], []

            # 👇 슬라이싱(table.rows[1:]) 절대 쓰지 마세요.
            row_list = list(table.rows)  # 안전하게 파이썬 리스트로 복제

            if len(row_list) == 0:
                return None

            # 첫 행을 헤더로
            first_row = row_list[0]
            headers = [cell.text.strip() for cell in first_row.cells]

            # 나머지는 데이터
            for r in row_list[1:]:
                rows.append([cell.text.strip() for cell in r.cells])

            table_element = PageElement(
                element_id=generate_element_id(page_id, ElementType.TABLE, element_index),
                element_type=ElementType.TABLE,
                table_data=TableElement(
                    element_id=generate_element_id(page_id, ElementType.TABLE, element_index),
                    headers=headers,
                    rows=rows,
                    extraction_confidence=0.8
                ),
                confidence=0.8
            )
            return table_element

        except Exception as e:
            logger.warning(f"PPT 표 추출 실패: {e!r}")  # repr로 에러 원인 더 명확히
            return None
    
    def _emu_to_px(self, emu_val: Any, dpi: int = 96) -> int:
        # 1 inch = 914400 EMU, 96 px/inch
        try:
            emu = int(emu_val)
        except Exception:
            emu = int(float(emu_val))
        px = int(round(emu * dpi / 914400))
        return max(0, px)  # 음수 방지

    def _shape_to_bbox(self, shape) -> BoundingBox:
        x = self._emu_to_px(shape.left)
        y = self._emu_to_px(shape.top)
        w = max(1, self._emu_to_px(shape.width))   # 0 방지
        h = max(1, self._emu_to_px(shape.height))  # 0 방지
        return BoundingBox(x=x, y=y, width=w, height=h)
    
    def _extract_slide_text(self, slide) -> str:
        """슬라이드에서 모든 텍스트 추출"""
        texts = []
        
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                texts.append(shape.text.strip())
        
        return "\n".join(texts)
    
    def _extract_title_from_text(self, text: str) -> Optional[str]:
        """텍스트에서 제목 추출"""
        if not text:
            return None
        
        lines = text.strip().split('\n')
        first_line = lines[0].strip()
        
        if 5 <= len(first_line) <= 100:
            return first_line
        
        return None
    
    async def _create_multimodal_index(self, doc_id: str, pages: List[PageInfo]):
        """멀티모달 요소를 포함한 LlamaIndex 생성"""
        logger.info(f"멀티모달 인덱스 생성 중: {doc_id}")
        
        documents = []
        
        for page in pages:
            # 각 요소를 별도 문서로 인덱싱
            for element in page.elements:
                text_content = ""
                
                if element.element_type == ElementType.TEXT:
                    text_content = element.text_content or ""
                
                elif element.element_type == ElementType.IMAGE:
                    # 이미지의 OCR 텍스트와 설명 결합
                    parts = []
                    if element.image_data and element.image_data.ocr_text:
                        parts.append(f"[OCR] {element.image_data.ocr_text}")
                    if element.image_data and element.image_data.description:
                        parts.append(f"[설명] {element.image_data.description}")
                    text_content = " ".join(parts)
                
                elif element.element_type == ElementType.TABLE:
                    # 표 데이터를 텍스트로 변환
                    if element.table_data:
                        parts = []
                        if element.table_data.headers:
                            parts.append(f"[헤더] {' | '.join(element.table_data.headers)}")
                        for row in element.table_data.rows:
                            parts.append(f"[행] {' | '.join(row)}")
                        text_content = "\n".join(parts)
                
                elif element.element_type == ElementType.CHART:
                    # 차트 설명
                    if element.chart_data and element.chart_data.description:
                        text_content = f"[차트] {element.chart_data.description}"
                
                if text_content.strip():
                    doc = Document(
                        text=text_content,
                        metadata={
                            "doc_id": doc_id,
                            "page_id": page.page_id,
                            "page_number": page.page_number,
                            "element_id": element.element_id,
                            "element_type": element.element_type.value,
                            "confidence": element.confidence
                        }
                    )
                    documents.append(doc)
        
        if not documents:
            logger.warning(f"인덱싱할 콘텐츠가 없습니다: {doc_id}")
            return
        
        try:
            # 멀티모달 벡터 인덱스 생성
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=self.embeddings,
                transformations=[self.text_splitter]
            )
            
            self.indexes[doc_id] = index
            logger.info(f"멀티모달 인덱스 생성 완료: {doc_id} ({len(documents)} 요소)")
            
        except Exception as e:
            logger.error(f"인덱스 생성 실패: {str(e)}")
            raise
    
    # 기존 DocumentAgent의 메서드들 유지
    def get_document(self, doc_id: str) -> Optional[DocumentMeta]:
        """문서 메타데이터 조회"""
        return self.documents.get(doc_id)
    
    def get_pages(self, doc_id: str) -> List[PageInfo]:
        """문서의 페이지 목록 조회"""
        return self.pages.get(doc_id, [])
    
    def get_index(self, doc_id: str) -> Optional[VectorStoreIndex]:
        """문서의 벡터 인덱스 조회"""
        return self.indexes.get(doc_id)
    
    def search_in_document(self, doc_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """멀티모달 문서 내 검색"""
        index = self.get_index(doc_id)
        if not index:
            return []
        
        try:
            # 쿼리 엔진 생성
            query_engine = index.as_query_engine(similarity_top_k=top_k)
            
            # 검색 수행
            response = query_engine.query(query)
            
            # 결과 정리
            results = []
            for node in response.source_nodes:
                results.append({
                    "text": node.text,
                    "score": node.score,
                    "metadata": node.metadata,
                    "page_id": node.metadata.get("page_id"),
                    "page_number": node.metadata.get("page_number"),
                    "element_id": node.metadata.get("element_id"),
                    "element_type": node.metadata.get("element_type"),
                    "confidence": node.metadata.get("confidence")
                })
            
            return results
            
        except Exception as e:
            logger.error(f"멀티모달 문서 검색 실패: {str(e)}")
            return []
    
    def search_by_element_type(self, doc_id: str, element_type: ElementType, query: str = None) -> List[PageElement]:
        """요소 타입별 검색"""
        pages = self.get_pages(doc_id)
        if not pages:
            return []
        
        results = []
        for page in pages:
            for element in page.elements:
                if element.element_type == element_type:
                    # 쿼리가 있는 경우 텍스트 매칭
                    if query:
                        element_text = ""
                        if element.element_type == ElementType.TEXT:
                            element_text = element.text_content or ""
                        elif element.element_type == ElementType.IMAGE and element.image_data:
                            element_text = f"{element.image_data.ocr_text or ''} {element.image_data.description or ''}"
                        elif element.element_type == ElementType.TABLE and element.table_data:
                            element_text = f"{' '.join(element.table_data.headers)} {' '.join([' '.join(row) for row in element.table_data.rows])}"
                        elif element.element_type == ElementType.CHART and element.chart_data:
                            element_text = element.chart_data.description or ""
                        
                        if query.lower() in element_text.lower():
                            results.append(element)
                    else:
                        results.append(element)
        
        return results
    
    def get_multimodal_stats(self, doc_id: str) -> Dict[str, Any]:
        """멀티모달 문서 통계"""
        doc_meta = self.get_document(doc_id)
        pages = self.get_pages(doc_id)
        
        if not doc_meta or not pages:
            return {}
        
        # 요소별 통계 계산
        element_stats = {
            "text": 0,
            "image": 0,
            "table": 0,
            "chart": 0,
            "total": 0
        }
        
        total_words = 0
        total_chars = 0
        ocr_text_count = 0
        ai_descriptions = 0
        
        for page in pages:
            total_words += page.word_count
            total_chars += len(page.raw_text)
            
            for element in page.elements:
                element_stats["total"] += 1
                
                if element.element_type == ElementType.TEXT:
                    element_stats["text"] += 1
                elif element.element_type == ElementType.IMAGE:
                    element_stats["image"] += 1
                    if element.image_data:
                        if element.image_data.ocr_text:
                            ocr_text_count += 1
                        if element.image_data.description:
                            ai_descriptions += 1
                elif element.element_type == ElementType.TABLE:
                    element_stats["table"] += 1
                elif element.element_type == ElementType.CHART:
                    element_stats["chart"] += 1
        
        return {
            "doc_id": doc_id,
            "title": doc_meta.title,
            "total_pages": len(pages),
            "total_words": total_words,
            "total_chars": total_chars,
            "avg_words_per_page": total_words / len(pages) if pages else 0,
            "has_index": doc_id in self.indexes,
            "elements": element_stats,
            "multimodal_features": {
                "ocr_enabled": self.enable_ocr,
                "vision_analysis_enabled": self.enable_vision_analysis,
                "ocr_text_extracted": ocr_text_count,
                "ai_descriptions_generated": ai_descriptions
            }
        }
    
    def list_documents(self) -> List[DocumentMeta]:
        """모든 문서 목록 조회"""
        return list(self.documents.values())
    
    async def get_page_summary(self, doc_id: str, page_id: str) -> Dict[str, Any]:
        """페이지 요약 정보"""
        pages = self.get_pages(doc_id)
        target_page = None
        
        for page in pages:
            if page.page_id == page_id:
                target_page = page
                break
        
        if not target_page:
            return {}
        
        # 요소별 요약
        summary = {
            "page_id": page_id,
            "page_number": target_page.page_number,
            "word_count": target_page.word_count,
            "element_counts": {
                "text": target_page.text_count,
                "image": target_page.image_count,
                "table": target_page.table_count,
                "chart": target_page.chart_count
            },
            "elements": []
        }
        
        # 각 요소의 요약 정보
        for element in target_page.elements:
            element_summary = {
                "element_id": element.element_id,
                "element_type": element.element_type.value,
                "confidence": element.confidence
            }
            
            if element.element_type == ElementType.TEXT:
                element_summary["preview"] = (element.text_content or "")[:100]
            elif element.element_type == ElementType.IMAGE and element.image_data:
                element_summary["description"] = element.image_data.description
                element_summary["has_ocr"] = bool(element.image_data.ocr_text)
            elif element.element_type == ElementType.TABLE and element.table_data:
                element_summary["dimensions"] = f"{element.table_data.row_count}x{element.table_data.col_count}"
                element_summary["has_headers"] = element.table_data.has_headers
            elif element.element_type == ElementType.CHART and element.chart_data:
                element_summary["chart_type"] = element.chart_data.chart_type
                element_summary["description"] = element.chart_data.description
            
            summary["elements"].append(element_summary)
        
        return summary


# 테스트 함수
async def test_multimodal_agent():
    """MultimodalDocumentAgent 테스트"""
    print("🧪 MultimodalDocumentAgent 테스트 시작...")
    
    # 에이전트 생성
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("⚠️  OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   멀티모달 기능 일부가 제한됩니다.")
    
    agent = MultimodalDocumentAgent(
        openai_api_key=api_key,
        enable_ocr=True,
        enable_vision_analysis=bool(api_key)
    )
    
    # 테스트 파일들
    test_files = [
        "Ch01_intro.pdf",
        "Ch01_intro.pptx"
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            try:
                print(f"\n📄 멀티모달 파싱 테스트: {test_file}")
                
                # 문서 파싱
                doc_meta = await agent.parse_document(test_file)
                
                print(f"✅ 파싱 성공!")
                print(f"   문서 ID: {doc_meta.doc_id}")
                print(f"   제목: {doc_meta.title}")
                print(f"   페이지 수: {doc_meta.total_pages}")
                print(f"   이미지: {doc_meta.total_images}개")
                print(f"   표: {doc_meta.total_tables}개")
                print(f"   차트: {doc_meta.total_charts}개")
                
                # 멀티모달 통계
                stats = agent.get_multimodal_stats(doc_meta.doc_id)
                print(f"   총 요소: {stats['elements']['total']}개")
                print(f"   OCR 추출: {stats['multimodal_features']['ocr_text_extracted']}개")
                print(f"   AI 설명: {stats['multimodal_features']['ai_descriptions_generated']}개")
                
                # 요소 타입별 검색 테스트
                images = agent.search_by_element_type(doc_meta.doc_id, ElementType.IMAGE)
                tables = agent.search_by_element_type(doc_meta.doc_id, ElementType.TABLE)
                print(f"   이미지 검색: {len(images)}개 발견")
                print(f"   표 검색: {len(tables)}개 발견")
                
                # 첫 번째 페이지 요약
                if stats['total_pages'] > 0:
                    pages = agent.get_pages(doc_meta.doc_id)
                    if pages:
                        page_summary = await agent.get_page_summary(doc_meta.doc_id, pages[0].page_id)
                        print(f"   첫 페이지 요소: {len(page_summary.get('elements', []))}개")
                
                # 멀티모달 검색 테스트
                if api_key and stats.get('has_index'):
                    search_results = agent.search_in_document(
                        doc_meta.doc_id,
                        "이미지",
                        top_k=3
                    )
                    print(f"   멀티모달 검색 결과: {len(search_results)}개")
                    
                    for result in search_results[:2]:  # 처음 2개만 출력
                        print(f"     - {result['element_type']}: {result['text'][:50]}...")
                
            except Exception as e:
                print(f"❌ 파싱 실패: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⏭️  테스트 파일 없음: {test_file}")
    
    # 문서 목록 출력
    docs = agent.list_documents()
    print(f"\n📚 총 {len(docs)}개 멀티모달 문서 로드됨")
    
    print("🎉 MultimodalDocumentAgent 테스트 완료!")


if __name__ == "__main__":
    asyncio.run(test_multimodal_agent())