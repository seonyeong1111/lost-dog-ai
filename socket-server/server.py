import socket
import cv2
import pickle
import struct
from ultralytics import YOLO
import os
import firebase_admin
from firebase_admin import credentials, storage, db
import time
import threading 
import sys

# Firebase 초기화 
cred = credentials.Certificate("dongmul-firebase-adminsdk-fbsvc-02ae7789b0.json")  
firebase_admin.initialize_app(cred, {
    'storageBucket': 'dongmul.firebasestorage.app',  
    'databaseURL': 'https://dongmul-default-rtdb.firebaseio.com/'  
})

def upload_image_to_firebase(image_buffer, label):
    """
    Firebase Storage에 이미지 업로드 (파일명: 라벨명.jpg, 덮어쓰기)
    
    Args:
        image_buffer (np.ndarray): JPEG 인코딩된 이미지 버퍼
        label (str): 라벨명 (예: 'target1')
    
    Returns:
        str: 업로드된 이미지의 공개 URL
    """
    bucket = storage.bucket()
    filename = f"detected_{label}.jpg"  
    blob = bucket.blob(filename)
    blob.upload_from_string(image_buffer.tobytes(), content_type='image/jpeg')
    url = blob.public_url
    print(f"✅ 이미지 Firebase Storage에 업로드 완료: {url}")
    return url

def update_realtime_db(label, findURL):
    ref = db.reference(f'animalList/{label}')
    update_data = {
        'find': 1,
        'findGPS': '울산광역시 남구 대학로 93',
        'findURL': findURL
    }
    ref.update(update_data)
    print(f"✅ Realtime DB 업데이트 완료: {label} -> {update_data}")

# 클라이언트 처리 함수 분리
def handle_client(client_socket, client_address, model):
    print(f"🔗 클라이언트 연결됨: {client_address}")
    frame_count = 0
    uploaded_to_db = set()  # 클라이언트별로 따로 관리

    try:
        while True:
            raw_msglen = recv_all(client_socket, 4)
            if not raw_msglen:
                print(f"📡 클라이언트 {client_address} 연결 종료")
                break

            msglen = struct.unpack('>I', raw_msglen)[0]
            raw_data = recv_all(client_socket, msglen)
            if not raw_data:
                print(f"📡 클라이언트 {client_address} 데이터 수신 실패")
                break

            img_data = pickle.loads(raw_data)
            frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

            if frame is not None:
                frame_count += 1
                results = model.predict(frame, conf=0.25, verbose=False)
                annotated_frame = results[0].plot()

                detections = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        detections.append({
                            'class': results[0].names[int(box.cls[0])],
                            'confidence': float(box.conf[0])
                        })

                if detections:
                    for det in detections:
                        label = det['class']
                        confidence = det['confidence']

                        if can_upload(label, cooldown=30):
                            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            url = upload_image_to_firebase(buffer, label)
                            update_realtime_db(label=label, findURL=url)

                        if label not in uploaded_to_db:
                            update_realtime_db(label=label, findURL=url)
                            uploaded_to_db.add(label)

                    det_info = ", ".join([f"{d['class']}({d['confidence']:.2f})" for d in detections])
                    print(f"📊 프레임 {frame_count}: {len(detections)}개 감지 - {det_info}")
                else:
                    if frame_count % 30 == 0:
                        print(f"📊 프레임 {frame_count}: 감지 없음")

                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                data = pickle.dumps(buffer)
                message_size = struct.pack('>I', len(data))
                client_socket.sendall(message_size + data)
    except Exception as e:
        print(f"❌ 처리 오류 ({client_address}): {e}")
    finally:
        client_socket.close()
        print(f"🛑 클라이언트 연결 종료됨: {client_address}")

# =====================================
# 서버 (추론을 수행하는 컴퓨터)
# =====================================

def run_server(host='0.0.0.0', port=8888, model_path = '../flask-backend/animal_detection/train/weights/best.pt'):
    uploaded_to_db = set()
    """추론 서버 실행"""
    
    print("🖥️  추론 서버 시작")
    print(f"모델 로딩 중: {model_path}")
    
    # 모델 파일 확인 및 로드
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        print("기본 YOLOv8s 모델을 사용합니다...")
        model_path = '../flask-backend/yolov8s.pt'
    
    try:
        model = YOLO(model_path)
        print("✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 서버 소켓 생성
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.settimeout(1.0)  
    running = True  

    try:
        server_socket.bind((host, port))
        server_socket.listen(5) # 동시 접속 허용 수

        print(f"📡 서버 대기 중: {host}:{port}")
        print("클라이언트 연결을 기다리는 중...")
        print("Ctrl+C로 서버 종료")

        while running:
            try:
                client_socket, client_address = server_socket.accept()  # 타임아웃 1초로 대기
                thread = threading.Thread(target=handle_client, args=(client_socket, client_address, model))
                thread.daemon = True  # 백그라운드 스레드로 실행
                thread.start()
            except socket.timeout:
                # 1초마다 여기로 오고, 종료 체크 가능
                pass
    except KeyboardInterrupt:
        print("\n🛑 서버 종료 중...")
        running = False
    except Exception as e:
        print(f"❌ 서버 오류: {e}")
    finally:
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

last_upload_time = {}

def can_upload(label, cooldown=10):
    """
    label 기준으로 cooldown 초 경과 여부 확인
    """
    now = time.time()
    if label not in last_upload_time or (now - last_upload_time[label]) > cooldown:
        last_upload_time[label] = now
        return True
    return False
