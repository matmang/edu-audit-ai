#!/usr/bin/env python3
"""
EDU-Audit 서버 테스트 스크립트
DocumentAgent 기능을 단계별로 테스트
"""

import asyncio
import aiohttp
import json
from pathlib import Path
import time

BASE_URL = "http://localhost:8000"

class ServerTester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health(self):
        """헬스 체크 테스트"""
        print("🔍 헬스 체크 테스트...")
        
        try:
            async with self.session.get(f"{self.base_url}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ 서버 상태: {data['status']}")
                    print(f"   버전: {data['version']}")
                    print(f"   준비 상태: {data['ready']}")
                    print(f"   등록된 에이전트: {data['agents']}")
                    return True
                else:
                    print(f"❌ 헬스 체크 실패: {resp.status}")
                    return False
        except Exception as e:
            print(f"❌ 서버 연결 실패: {str(e)}")
            return False
    
    async def test_agents_list(self):
        """에이전트 목록 테스트"""
        print("\n🤖 에이전트 목록 테스트...")
        
        try:
            async with self.session.get(f"{self.base_url}/agents") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ 총 에이전트: {data['total_agents']}개")
                    
                    for name, info in data['agents'].items():
                        print(f"   - {name}: {info['status']} ({info['agent_type']})")
                        print(f"     지원 액션: {info['supported_actions']}")
                    return True
                else:
                    print(f"❌ 에이전트 목록 조회 실패: {resp.status}")
                    return False
        except Exception as e:
            print(f"❌ 에이전트 목록 조회 오류: {str(e)}")
            return False
    
    async def test_document_upload(self, test_file_path: str = None):
        """문서 업로드 테스트"""
        print("\n📄 문서 업로드 테스트...")
        
        # 테스트 파일 찾기
        if not test_file_path:
            test_files = [
                "sample_docs/sample.pdf",
            ]
            
            for file_path in test_files:
                if Path(file_path).exists():
                    test_file_path = file_path
                    break
        
        if not test_file_path or not Path(test_file_path).exists():
            print("❌ 테스트용 PDF 파일을 찾을 수 없습니다.")
            print("   다음 중 하나를 준비해주세요: sample_docs/sample.pdf, test.pdf, sample.pdf")
            return False
        
        print(f"📁 테스트 파일: {test_file_path}")
        
        try:
            data = aiohttp.FormData()
            f = open(test_file_path, 'rb')
            data.add_field(
                'file',
                f,
                filename=Path(test_file_path).name,
                content_type='application/pdf'
            )
            
            print("⏳ 업로드 중... (시간이 걸릴 수 있습니다)")
            start_time = time.time()
            
            async with self.session.post(f"{self.base_url}/document/upload", data=data) as resp:
                processing_time = time.time() - start_time
                
                if resp.status == 200:
                    result = await resp.json()
                    print(f"✅ 문서 업로드 성공!")
                    print(f"   처리 시간: {processing_time:.1f}초")
                    print(f"   문서 ID: {result['doc_meta']['doc_id']}")
                    print(f"   제목: {result['doc_meta']['title']}")
                    print(f"   페이지 수: {result['doc_meta']['total_pages']}")
                    return result['doc_meta']['doc_id']
                else:
                    error = await resp.text()
                    print(f"❌ 문서 업로드 실패: {resp.status}")
                    print(f"   오류: {error}")
                    return None
                    
        except Exception as e:
            print(f"❌ 문서 업로드 오류: {str(e)}")
            return None
    
    async def test_document_info(self, doc_id: str):
        """문서 정보 조회 테스트"""
        print(f"\n📋 문서 정보 조회 테스트 (doc_id: {doc_id[:8]}...)...")
        
        try:
            async with self.session.get(f"{self.base_url}/document/{doc_id}/info") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    doc_meta = data['doc_meta']
                    print(f"✅ 문서 정보 조회 성공!")
                    print(f"   제목: {doc_meta['title']}")
                    print(f"   타입: {doc_meta['doc_type']}")
                    print(f"   생성일: {doc_meta['created_at'][:19]}")
                    return True
                else:
                    error = await resp.text()
                    print(f"❌ 문서 정보 조회 실패: {resp.status} - {error}")
                    return False
        except Exception as e:
            print(f"❌ 문서 정보 조회 오류: {str(e)}")
            return False
    
    async def test_document_stats(self, doc_id: str):
        """문서 통계 조회 테스트"""
        print(f"\n📊 문서 통계 조회 테스트...")
        
        try:
            async with self.session.get(f"{self.base_url}/document/{doc_id}/stats") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    stats = data['stats']
                    print(f"✅ 문서 통계 조회 성공!")
                    print(f"   총 슬라이드: {stats['total_slides']}")
                    print(f"   캡션 생성률: {stats['caption_coverage']:.1%}")
                    print(f"   평균 캡션 길이: {stats['avg_caption_length']:.0f}자")
                    print(f"   총 이미지 크기: {stats['total_image_size_mb']:.1f}MB")
                    return True
                else:
                    error = await resp.text()
                    print(f"❌ 문서 통계 조회 실패: {resp.status} - {error}")
                    return False
        except Exception as e:
            print(f"❌ 문서 통계 조회 오류: {str(e)}")
            return False
    
    async def test_document_slides(self, doc_id: str):
        """슬라이드 조회 테스트"""
        print(f"\n🎞️ 슬라이드 조회 테스트...")
        
        try:
            async with self.session.get(f"{self.base_url}/document/{doc_id}/slides") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    slides = data['slides']
                    print(f"✅ 슬라이드 조회 성공!")
                    print(f"   총 슬라이드: {data['total_slides']}개")
                    
                    # 첫 번째 슬라이드 정보 출력
                    if slides:
                        first_slide = slides[0]
                        print(f"   첫 슬라이드: {first_slide['page_id']}")
                        print(f"   캡션: {first_slide['caption'][:100]}...")
                        print(f"   이미지: {'있음' if first_slide['has_image'] else '없음'}")
                    
                    return True
                else:
                    error = await resp.text()
                    print(f"❌ 슬라이드 조회 실패: {resp.status} - {error}")
                    return False
        except Exception as e:
            print(f"❌ 슬라이드 조회 오류: {str(e)}")
            return False
    
    async def test_document_search(self, doc_id: str):
        """문서 검색 테스트"""
        print(f"\n🔍 문서 검색 테스트...")
        
        # 여러 검색어로 테스트
        test_queries = [
            "머신러닝",
            "개요",
            "학습"
        ]
        
        for query in test_queries:
            try:
                search_data = {"query": query, "top_k": 3}
                
                async with self.session.post(
                    f"{self.base_url}/document/{doc_id}/search",
                    json=search_data
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data['results']
                        print(f"✅ 검색 성공 ('{query}'): {data['total_results']}개 결과")
                        
                        # 상위 결과 출력
                        for i, result in enumerate(results[:2], 1):
                            page_id = result.get('page_id', 'Unknown')
                            text_preview = result.get('text', '')[:80]
                            print(f"   {i}. {page_id}: {text_preview}...")
                        
                    else:
                        error = await resp.text()
                        print(f"❌ 검색 실패 ('{query}'): {resp.status} - {error}")
                        
            except Exception as e:
                print(f"❌ 검색 오류 ('{query}'): {str(e)}")
        
        return True
    
    async def test_document_list(self):
        """문서 목록 조회 테스트"""
        print(f"\n📚 문서 목록 조회 테스트...")
        
        try:
            async with self.session.get(f"{self.base_url}/document/list") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    documents = data['documents']
                    print(f"✅ 문서 목록 조회 성공!")
                    print(f"   총 문서: {data['total_documents']}개")
                    
                    for doc in documents:
                        print(f"   - {doc['title']} ({doc['doc_id'][:8]}..., {doc['total_pages']}페이지)")
                    
                    return True
                else:
                    error = await resp.text()
                    print(f"❌ 문서 목록 조회 실패: {resp.status} - {error}")
                    return False
        except Exception as e:
            print(f"❌ 문서 목록 조회 오류: {str(e)}")
            return False
    
    async def test_server_stats(self):
        """서버 통계 테스트"""
        print(f"\n📈 서버 통계 테스트...")
        
        try:
            async with self.session.get(f"{self.base_url}/stats") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    execution_stats = data['execution_stats']
                    registry_stats = data['registry_stats']
                    
                    print(f"✅ 서버 통계 조회 성공!")
                    print(f"   업타임: {data['uptime_seconds']:.0f}초")
                    print(f"   총 요청: {execution_stats['total_requests']}")
                    print(f"   성공 요청: {execution_stats['successful_requests']}")
                    print(f"   실패 요청: {execution_stats['failed_requests']}")
                    print(f"   처리된 문서: {execution_stats['documents_processed']}")
                    print(f"   등록된 에이전트: {registry_stats['total_agents']}개")
                    
                    return True
                else:
                    error = await resp.text()
                    print(f"❌ 서버 통계 조회 실패: {resp.status} - {error}")
                    return False
        except Exception as e:
            print(f"❌ 서버 통계 조회 오류: {str(e)}")
            return False
    
    async def test_quality_analysis(self, doc_id: str):
        """품질 분석 테스트"""
        print(f"\n🔍 품질 분석 테스트...")
        
        try:
            print("⏳ 품질 분석 중... (Vision 모델 사용으로 시간이 걸릴 수 있습니다)")
            start_time = time.time()
            
            async with self.session.post(f"{self.base_url}/document/{doc_id}/analyze/quality", timeout=aiohttp.ClientTimeout(total=600)) as resp:
                processing_time = time.time() - start_time
                
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ 품질 분석 성공! (처리 시간: {processing_time:.1f}초)")
                    print(f"   보고서 ID: {data['report_id']}")
                    print(f"   총 이슈: {data['total_issues']}개")
                    print(f"   분석 타입: {data['analysis_type']}")
                    
                    # 이슈 상세 출력 (상위 3개만)
                    issues = data.get('issues', [])
                    if issues:
                        print(f"   주요 이슈들:")
                        for i, issue in enumerate(issues[:3], 1):
                            issue_type = issue.get('issue_type', 'unknown')
                            message = issue.get('message', '')[:80]
                            confidence = issue.get('confidence', 0)
                            print(f"     {i}. [{issue_type.upper()}] {message}... (신뢰도: {confidence:.2f})")
                    
                    return True
                    
                elif resp.status == 503:
                    print("❌ QualityAgent가 등록되지 않았습니다.")
                    return False
                else:
                    error = await resp.text()
                    print(f"❌ 품질 분석 실패: {resp.status} - {error}")
                    return False
                    
        except Exception as e:
            print(f"❌ 품질 분석 오류: {str(e)}")
            return False

    async def test_quality_summary(self, doc_id: str):
        """품질 요약 테스트"""
        print(f"\n📊 품질 요약 테스트...")
        
        try:
            async with self.session.get(f"{self.base_url}/document/{doc_id}/summary/quality") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    summary = data.get('summary', {})
                    print(f"✅ 품질 요약 조회 성공!")
                    print(f"   품질 점수: {summary.get('quality_score', 0):.2f}/1.0")
                    print(f"   총 이슈: {summary.get('total_issues', 0)}개")
                    
                    # 이슈 타입별 분포
                    by_type = summary.get('by_type', {})
                    if by_type:
                        print(f"   이슈 타입별 분포:")
                        for issue_type, count in by_type.items():
                            print(f"     - {issue_type}: {count}개")
                    
                    # 권장사항
                    recommendations = summary.get('recommendations', [])
                    if recommendations:
                        print(f"   권장사항:")
                        for rec in recommendations[:3]:
                            print(f"     - {rec}")
                    
                    return True
                    
                elif resp.status == 503:
                    print("❌ QualityAgent가 등록되지 않았습니다.")
                    return False
                else:
                    error = await resp.text()
                    print(f"❌ 품질 요약 조회 실패: {resp.status} - {error}")
                    return False
                    
        except Exception as e:
            print(f"❌ 품질 요약 조회 오류: {str(e)}")
            return False
    
    async def test_quality_config(self):
        """QualityAgent 설정 조회 테스트"""
        print(f"\n⚙️ QualityAgent 설정 테스트...")
        
        try:
            async with self.session.get(f"{self.base_url}/quality/config") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    config = data.get('config', {})
                    print(f"✅ QualityAgent 설정 조회 성공!")
                    print(f"   슬라이드당 최대 이슈: {config.get('max_issues_per_slide', 'N/A')}개")
                    print(f"   신뢰도 임계값: {config.get('confidence_threshold', 'N/A')}")
                    print(f"   심각도 필터: {config.get('issue_severity_filter', 'N/A')}")
                    print(f"   Vision 분석 활성화: {config.get('enable_vision_analysis', 'N/A')}")
                    
                    return True
                    
                elif resp.status == 503:
                    print("❌ QualityAgent가 등록되지 않았습니다.")
                    return False
                else:
                    error = await resp.text()
                    print(f"❌ QualityAgent 설정 조회 실패: {resp.status} - {error}")
                    return False
                    
        except Exception as e:
            print(f"❌ QualityAgent 설정 조회 오류: {str(e)}")
            return False


async def run_full_test(test_file_path: str = None):
    """전체 테스트 스위트 실행"""
    print("🧪 EDU-Audit 서버 테스트 시작")
    print("=" * 50)
    
    async with ServerTester() as tester:
        # 1. 기본 연결 테스트
        if not await tester.test_health():
            print("\n❌ 서버가 실행되지 않았거나 응답하지 않습니다.")
            print("   다음 명령어로 서버를 시작하세요:")
            print("   python -m app.main")
            return False
        
        # 2. 에이전트 확인
        await tester.test_agents_list()
        
        # 3. 문서 업로드 (메인 테스트)
        doc_id = await tester.test_document_upload(test_file_path)
        if not doc_id:
            print("\n❌ 문서 업로드 실패로 인해 나머지 테스트를 진행할 수 없습니다.")
            return False
        
        # 4. 문서 관련 모든 기능 테스트
        await tester.test_document_info(doc_id)
        await tester.test_document_stats(doc_id)
        await tester.test_document_slides(doc_id)
        await tester.test_document_search(doc_id)
        
        # 5. 목록 및 통계 테스트
        await tester.test_document_list()
        await tester.test_server_stats()

        # 6. 문서 품질 검수 기능 테스트
        await tester.test_quality_analysis(doc_id)
        await tester.test_quality_summary(doc_id)
        await tester.test_quality_config()
    
    print("\n" + "=" * 50)
    print("🎉 테스트 완료!")
    return True


async def run_quick_test():
    """빠른 연결 테스트"""
    print("🚀 빠른 연결 테스트...")
    
    async with ServerTester() as tester:
        health_ok = await tester.test_health()
        if health_ok:
            await tester.test_agents_list()
            print("✅ 서버가 정상적으로 실행되고 있습니다.")
        return health_ok


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            # 빠른 테스트
            asyncio.run(run_quick_test())
        elif sys.argv[1] == "full":
            # 전체 테스트 (파일 경로 선택사항)
            test_file = sys.argv[2] if len(sys.argv) > 2 else None
            asyncio.run(run_full_test(test_file))
        else:
            print("사용법:")
            print("  python test_server.py quick          # 빠른 연결 테스트")
            print("  python test_server.py full [파일]    # 전체 기능 테스트")
    else:
        # 기본값: 전체 테스트
        asyncio.run(run_full_test())