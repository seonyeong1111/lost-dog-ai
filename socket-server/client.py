import socket
import cv2
import pickle
import struct

def run_client(server_ip, server_port=8888, camera_index=0):
    """웹캠 클라이언트 실행"""
    
    print("📹 웹캠 클라이언트 시작")
    print(f"서버 연결 중: {server_ip}:{server_port}")
    
    # IP 주소 유효성 검사
    if not is_valid_ip(server_ip):
        print(f"❌ 잘못된 IP 주소 형식: {server_ip}")
        print("올바른 형식: 192.168.1.100 또는 localhost")
        return
    
    # 서버에 연결
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(10)  # 10초 타임아웃
    
    try:
        client_socket.connect((server_ip, server_port))
        client_socket.settimeout(None)  # 타임아웃 해제
        print("✅ 서버 연결 성공")
        
        # 웹캠 초기화
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ 웹캠({camera_index})을 열 수 없습니다")
            print("다른 카메라 인덱스를 시도해보세요: 0, 1, 2...")
            return
        
        # 웹캠 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        print("📡 스트리밍 시작")
        print("ESC 키를 눌러 종료하세요")
        
        frame_count = 0
        
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                print("❌ 프레임을 읽을 수 없습니다")
                break
            
            frame_count += 1
            
            # 이미지 압축
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            
            try:
                # 서버로 전송
                data = pickle.dumps(buffer)
                message_size = struct.pack('>I', len(data))
                client_socket.sendall(message_size + data)
                
                # 결과 받기
                raw_msglen = recv_all(client_socket, 4)
                if not raw_msglen:
                    print("📡 서버와의 연결이 끊어졌습니다")
                    break
                
                msglen = struct.unpack('>I', raw_msglen)[0]
                raw_data = recv_all(client_socket, msglen)
                if not raw_data:
                    print("📡 결과 데이터 수신 실패")
                    break
                
                # 결과 이미지 디코딩
                result_buffer = pickle.loads(raw_data)
                result_frame = cv2.imdecode(result_buffer, cv2.IMREAD_COLOR)
                
                if result_frame is not None:
                    # 결과 화면에 표시
                    cv2.imshow('YOLO 실시간 감지', result_frame)
                    
                    # 프레임 카운터 표시
                    if frame_count % 30 == 0:
                        print(f"📡 전송됨: {frame_count} 프레임")
                
            except Exception as e:
                print(f"❌ 통신 오류: {e}")
                break
            
            # ESC 키로 종료
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("🛑 ESC 키가 눌렸습니다. 종료합니다.")
                break
            elif key == ord('q'):  # Q 키
                print("🛑 Q 키가 눌렸습니다. 종료합니다.")
                break
                
    except socket.timeout:
        print("❌ 서버 연결 시간 초과")
        print("서버가 실행 중인지 확인하세요")
    except ConnectionRefusedError:
        print("❌ 서버에 연결할 수 없습니다")
        print("서버가 실행 중인지 확인하세요")
    except KeyboardInterrupt:
        print("\n🛑 클라이언트 종료 중...")
    except Exception as e:
        print(f"❌ 클라이언트 오류: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
        client_socket.close()
        cv2.destroyAllWindows()
        print("✅ 클라이언트 종료됨")

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