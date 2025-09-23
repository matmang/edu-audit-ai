"""
EDU-Audit MCP 서버 런처 및 테스트 도구
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any

# 런처 스크립트
class EDUAuditLauncher:
    """EDU-Audit MCP 서버 런처"""
    
    def __init__(self):
        self.config_file = Path("mcp_config.json")
        self.env_file = Path(".env")
    
    def check_dependencies(self) -> bool:
        """의존성 확인"""
        required_packages = [
            "mcp", "openai", "llama-index", "pdfplumber", 
            "python-pptx", "aiohttp", "pydantic", "python-dotenv"
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing.append(package)
        
        if missing:
            print(f"❌ 다음 패키지가 필요합니다: {', '.join(missing)}")
            print(f"설치 명령: pip install {' '.join(missing)}")
            return False
        
        print("✅ 모든 의존성이 설치되어 있습니다.")
        return True
    
    def check_environment(self) -> bool:
        """환경 변수 확인"""
        from dotenv import load_dotenv
        from pathlib import Path
        env_path = Path(__file__).resolve().parents[2] / '.env.dev'
        load_dotenv(env_path)
        
        openai_key = os.getenv("OPENAI_API_KEY")
        serpapi_key = os.getenv("SERPAPI_API_KEY")
        
        if not openai_key:
            print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            print("   .env 파일에 OPENAI_API_KEY=your_key_here 추가")
            return False
        
        print("✅ OpenAI API 키가 설정되어 있습니다.")
        
        if not serpapi_key:
            print("⚠️  SERPAPI_API_KEY가 설정되지 않았습니다.")
            print("   사실 검증 기능이 제한될 수 있습니다.")
        else:
            print("✅ SerpAPI 키도 설정되어 있습니다.")
        
        return True
    
    def generate_config(self):
        """MCP 설정 파일 생성"""
        config = {
            "mcpServers": {
                "edu-audit": {
                    "command": "python",
                    "args": ["-m", "src.mcp_server"],
                    "env": {
                        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
                        "SERPAPI_API_KEY": "${SERPAPI_API_KEY}"
                    }
                }
            }
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ MCP 설정 파일 생성: {self.config_file}")
        print("   Claude Desktop 설정에 이 내용을 추가하세요.")
    
    def test_server(self) -> bool:
        """서버 기본 테스트"""
        try:
            # 서버 모듈 import 테스트
            from src.mcp_server import EDUAuditMCPServer
            
            server = EDUAuditMCPServer()
            print("✅ MCP 서버 초기화 성공")
            
            # 에이전트 상태 확인
            if server.document_agent:
                print("✅ Document Agent 준비됨")
            if server.quality_agent:
                print("✅ Quality Agent 준비됨")
            if server.fact_agent:
                print("✅ Fact Check Agent 준비됨")
            
            return True
            
        except Exception as e:
            print(f"❌ 서버 테스트 실패: {str(e)}")
            return False
    
    def run_setup(self):
        """전체 설정 프로세스 실행"""
        print("🚀 EDU-Audit MCP 서버 설정을 시작합니다...\n")
        
        # 1. 의존성 확인
        print("1. 의존성 확인...")
        if not self.check_dependencies():
            return False
        
        # 2. 환경 변수 확인
        print("\n2. 환경 변수 확인...")
        if not self.check_environment():
            return False
        
        # 3. 서버 테스트
        print("\n3. 서버 테스트...")
        if not self.test_server():
            return False
        
        # 4. 설정 파일 생성
        print("\n4. 설정 파일 생성...")
        self.generate_config()
        
        print("\n🎉 설정이 완료되었습니다!")
        print("\n다음 단계:")
        print("1. Claude Desktop을 실행하세요")
        print(f"2. MCP 설정에 {self.config_file} 내용을 추가하세요")
        print("3. Claude Desktop을 재시작하세요")
        print("4. 'run_full_audit' 도구를 사용해보세요")
        
        return True


# 테스트 도구
class EDUAuditTester:
    """EDU-Audit MCP 도구 테스트"""
    
    def __init__(self):
        self.test_data_dir = Path("test_data")
        self.test_data_dir.mkdir(exist_ok=True)
    
    def create_test_document(self) -> str:
        """테스트용 더미 문서 생성"""
        # 간단한 텍스트 파일 생성 (실제 PDF/PPT 대신)
        test_content = """
        AI와 머신러닝 개요
        
        1. 머신러닝 소개
        머신러닝은 인공지능의 한 분야로, 컴퓨터가 명시적으로 프로그래밍되지 않고도 
        데이타를 통해 학습할 수 있는 능력을 제공합니다.
        
        2. 딥러닝과 심층학습
        딥러닝과 심층학습은 같은 개념입니다. 최신 연구에 따르면 딥러닝 모델의 
        정확도가 95%를 넘었다고 합니다.
        
        3. 알고리듬 종류
        - 지도학습 알고리듬
        - 비지도학습 알고리듬  
        - 강화학습 알고리듬
        
        4. 최신 동향
        GPT-4는 2023년에 출시되었으며, 현재 가장 강력한 언어 모델입니다.
        작년 AI 시장 규모는 100조원을 넘었습니다.
        """
        
        test_file = self.test_data_dir / "test_document.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        return str(test_file)
    
    async def test_tools_manually(self):
        """도구들을 직접 테스트"""
        from src.mcp.mcp_server import EDUAuditMCPServer
        
        print("🧪 EDU-Audit 도구 수동 테스트 시작...\n")
        
        server = EDUAuditMCPServer()
        
        # 테스트 문서 생성
        test_file = self.create_test_document()
        print(f"📄 테스트 문서 생성: {test_file}")
        
        # 1. check_typo 테스트
        print("\n1. 오탈자 검사 테스트...")
        try:
            typo_result = await server._handle_check_typo({
                "text": "딥러닝과 심층학습을 사용한 알고리듬 개발에서 데이타 전처리가 중요합니다.",
                "language": "ko"
            })
            print(f"   결과: {typo_result[0].text[:100]}...")
        except Exception as e:
            print(f"   ❌ 오류: {str(e)}")
        
        # 2. fact_check 테스트
        print("\n2. 사실 검증 테스트...")
        try:
            fact_result = await server._handle_fact_check({
                "sentence": "GPT-4는 2023년에 출시되었습니다",
                "context": "AI 모델 발전사"
            })
            print(f"   결과: {fact_result[0].text[:100]}...")
        except Exception as e:
            print(f"   ❌ 오류: {str(e)}")
        
        # 3. 전체 검수 테스트 (텍스트 파일로는 제한적)
        print("\n3. 문서 검수 시뮬레이션...")
        print("   (실제 PDF/PPT 파일이 있다면 run_full_audit 사용 가능)")
        
        # 4. 문서 목록 테스트
        print("\n4. 문서 목록 조회...")
        try:
            docs_result = await server._handle_list_documents({})
            print(f"   결과: {docs_result[0].text}")
        except Exception as e:
            print(f"   ❌ 오류: {str(e)}")
        
        print("\n🎉 수동 테스트 완료!")
    
    def test_with_real_file(self, file_path: str):
        """실제 파일로 전체 테스트"""
        if not Path(file_path).exists():
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            return
        
        print(f"📄 실제 파일 테스트: {file_path}")
        print("Claude Desktop에서 다음 명령을 사용하세요:")
        print(f'''
run_full_audit 도구 사용:
{{
  "file_path": "{file_path}",
  "include_fact_check": true,
  "quality_level": "standard"
}}
        ''')
    
    def generate_test_scenarios(self):
        """테스트 시나리오 생성"""
        scenarios = {
            "basic_audit": {
                "tool": "run_full_audit",
                "description": "기본 문서 검수",
                "params": {
                    "file_path": "./sample.pdf",
                    "quality_level": "standard"
                }
            },
            "fact_check_claim": {
                "tool": "fact_check", 
                "description": "특정 주장 검증",
                "params": {
                    "sentence": "한국의 인구는 약 5천만명입니다"
                }
            },
            "typo_check": {
                "tool": "check_typo",
                "description": "오탈자 검사",
                "params": {
                    "text": "머신러닝과 기계학습, 딥러닝과 심층학습의 차이점",
                    "language": "ko"
                }
            },
            "content_query": {
                "tool": "query_content",
                "description": "문서 내용 질의",
                "params": {
                    "question": "딥러닝의 주요 특징은 무엇인가요?"
                }
            }
        }
        
        scenarios_file = Path("test_scenarios.json")
        with open(scenarios_file, 'w', encoding='utf-8') as f:
            json.dump(scenarios, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 테스트 시나리오 생성: {scenarios_file}")
        return scenarios


def main():
    """메인 실행 함수"""
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python launcher.py setup     # 초기 설정")
        print("  python launcher.py test      # 수동 테스트")
        print("  python launcher.py test-file <파일경로>  # 실제 파일 테스트")
        print("  python launcher.py scenarios # 테스트 시나리오 생성")
        return
    
    command = sys.argv[1]
    
    if command == "setup":
        launcher = EDUAuditLauncher()
        launcher.run_setup()
    
    elif command == "test":
        tester = EDUAuditTester()
        asyncio.run(tester.test_tools_manually())
    
    elif command == "test-file":
        if len(sys.argv) < 3:
            print("파일 경로를 지정해주세요.")
            return
        
        tester = EDUAuditTester()
        tester.test_with_real_file(sys.argv[2])
    
    elif command == "scenarios":
        tester = EDUAuditTester()
        scenarios = tester.generate_test_scenarios()
        
        print("\n생성된 테스트 시나리오:")
        for name, scenario in scenarios.items():
            print(f"  {name}: {scenario['description']}")
    
    else:
        print(f"알 수 없는 명령: {command}")


if __name__ == "__main__":
    main()