import sys
from server import run_server
from client import run_client
import socket

def get_local_ip():
    """로컬 IP 주소 찾기"""
    try:
        # 임시 소켓으로 로컬 IP 확인
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "127.0.0.1"
    
def main():
    print("=" * 50)
    print("간단한 소켓 웹캠 추론 시스템")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("사용법:")
        print("  서버: python script.py server")
        print("  클라이언트: python script.py client <서버_IP>")
        print("\n예시:")
        print("  python script.py server")
        print("  python script.py client 192.168.1.100")
        print("  python script.py client localhost")
        print(f"\n현재 컴퓨터 IP: {get_local_ip()}")
        return
    
    mode = sys.argv[1].lower()
    
    if mode == "server":
        # 서버 실행
        local_ip = get_local_ip()
        print(f"🌐 서버 IP 주소: {local_ip}")
        print("클라이언트에서 이 IP로 연결하세요")

        #로컬용
        #model_path = '../flask-backend/app/animal_detection/train/weights/best.pt'
        #도커배포용
        #model_path = '/app/data/animal_detection/train/weights/best.pt'
        #임시용
        model_path = '../flask-backend/best.pt'

        run_server(model_path=model_path)
        
    elif mode == "client":
        if len(sys.argv) < 3:
            print("❌ 서버 IP를 입력해주세요")
            print("예시:")
            print("  python script.py client 192.168.1.100")
            print("  python script.py client localhost")
            return
        
        server_ip = sys.argv[2]
        
        # localhost를 127.0.0.1로 변환
        if server_ip.lower() == 'localhost':
            server_ip = '127.0.0.1'
        
        run_client(server_ip)
        
    else:
        print("❌ 'server' 또는 'client'를 입력해주세요")

if __name__ == "__main__":
    main()