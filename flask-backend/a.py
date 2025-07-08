# =====================================
# 수정된 간단한 소켓 웹캠 추론 시스템
# =====================================

import socket
import cv2
import numpy as np
import pickle
import struct
from ultralytics import YOLO
import sys
import os

# =====================================
# 서버 (추론을 수행하는 컴퓨터)
# =====================================

def run_server(host='0.0.0.0', port=8888, model_path='animal_detection/train/weights/best.pt'):
    """추론 서버 실행"""
    
    print("🖥️  추론 서버 시작")
    print(f"모델 로딩 중: {model_path}")
    
    # 모델 파일 확인 및 로드
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        print("기본 YOLOv8s 모델을 사용합니다...")
        model_path = 'yolov8s.pt'
    
    try:
        model = YOLO(model_path)
        print("✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 서버 소켓 생성
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((host, port))
        server_socket.listen(1)
        
        print(f"📡 서버 대기 중: {host}:{port}")
        print("클라이언트 연결을 기다리는 중...")
        print("Ctrl+C로 서버 종료")
        
        # 클라이언트 연결 대기
        client_socket, client_address = server_socket.accept()
        print(f"🔗 클라이언트 연결됨: {client_address}")
        
        frame_count = 0
        
        while True:
            try:
                # 이미지 데이터 크기 받기
                raw_msglen = recv_all(client_socket, 4)
                if not raw_msglen:
                    print("📡 클라이언트가 연결을 종료했습니다")
                    break
                
                msglen = struct.unpack('>I', raw_msglen)[0]
                
                # 이미지 데이터 받기
                raw_data = recv_all(client_socket, msglen)
                if not raw_data:
                    print("📡 데이터 수신 중단")
                    break
                
                # 이미지 디코딩
                img_data = pickle.loads(raw_data)
                frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    frame_count += 1
                    
                    # YOLO 추론
                    results = model.predict(frame, conf=0.25, verbose=False)
                    
                    # 결과를 이미지에 그리기
                    annotated_frame = results[0].plot()
                    
                    # 감지 정보 추출
                    detections = []
                    if results[0].boxes is not None:
                        for box in results[0].boxes:
                            detections.append({
                                'class': results[0].names[int(box.cls[0])],
                                'confidence': float(box.conf[0])
                            })
                    
                    # 결과 출력
                    if detections:
                        det_info = ", ".join([f"{d['class']}({d['confidence']:.2f})" for d in detections])
                        print(f"📊 프레임 {frame_count}: {len(detections)}개 감지 - {det_info}")
                    else:
                        if frame_count % 30 == 0:  # 30프레임마다 출력
                            print(f"📊 프레임 {frame_count}: 감지 없음")
                    
                    # 결과 이미지 인코딩
                    _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    
                    # 결과 전송
                    data = pickle.dumps(buffer)
                    message_size = struct.pack('>I', len(data))
                    client_socket.sendall(message_size + data)
                
            except Exception as e:
                print(f"❌ 처리 오류: {e}")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 서버 종료 중...")
    except Exception as e:
        print(f"❌ 서버 오류: {e}")
    finally:
        if 'client_socket' in locals():
            client_socket.close()
        server_socket.close()
        print("✅ 서버 종료됨")

# =====================================
# 유틸리티 함수
# =====================================

def recv_all(sock, n):
    """정확히 n바이트를 받을 때까지 수신"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def is_valid_ip(ip):
    """IP 주소 유효성 검사"""
    if ip in ['localhost', '127.0.0.1']:
        return True
    
    try:
        parts = ip.split('.')
        if len(parts) != 4:
            return False
        for part in parts:
            if not 0 <= int(part) <= 255:
                return False
        return True
    except:
        return False

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

# =====================================
# 메인 실행
# =====================================

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
        
        model_path = 'animal_detection/train/weights/best.pt'
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