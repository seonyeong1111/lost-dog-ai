from flask import Flask, request, jsonify
import requests
from pathlib import Path

app = Flask(__name__)

TRAIN_DIR = "./train_data"

#이미지 다운로드 함수
# ./train_data/target{targetId}/{1부터 차례대로}.jpg 구조로 저장하고 "./train_data/target{targetId}" 경로 반환
def download_images(target_id):
    base_url = f"https://dongmul-default-rtdb.firebaseio.com/animalList/target{target_id}/fileURL"
    # fileURL 버킷 안에 num 가져오기
    num_res = requests.get(f"{base_url}/num.json")
    if num_res.status_code != 200:
        return False, "Failed to get num"
    num_files = num_res.json()
    if not num_files or num_files == 0:
        return False, "No images to download"

    label_dir = Path(TRAIN_DIR) / f"target{target_id}"
    label_dir.mkdir(parents=True, exist_ok=True) #./train_data가 없어도 자동으로 생성, exist_ok=True: 이미 폴더가 존재하면 에러 없이 그냥 넘어감

    downloaded_count = 0

    for i in range(1, num_files + 1):
        url_res = requests.get(f"{base_url}/url{i}.json")
        if url_res.status_code != 200:
            print(f"Failed to get url{i}")
            continue
        url = url_res.json()
        try:
            img_res = requests.get(url) # ① 이미지 URL로 요청 보내기
            img_res.raise_for_status() # ② 응답 상태 확인 (오류 발생 시 예외)
            with open(label_dir / f"{i}.jpg", "wb") as f: # ③ 파일 열기 (바이너리 쓰기 모드)
                f.write(img_res.content) # ④ 이미지 내용 저장
            downloaded_count += 1
        except Exception as e:
            print(f"Error downloading image {url}: {e}")
    if downloaded_count == 0:
        return False, "All downloads failed"

    return True, str(label_dir)

# ⑥ 모델 학습 함수 << 코드 넣어주세요!
#어떤 데이터 필요하신지 몰라서 일단 이미지 경로만 가져왔습니다
################################################################

def train_model(data_path):
    # 임시 테스트 코드입니다
    print(f"📦 Training model with data in {data_path}")
    return True, "Model trained successfully"

#################################################################

#전체 파이프라인 함수
def run_training_pipeline(target_id):
    #1단계: 이미지 다운로드 함수
    success, result = download_images(target_id)
    if not success:
        return False, result

    data_path = result
    #2단계: 모델 학습 함수 코드 넣어주세요 파이팅~!
    success, msg = train_model(data_path)
    if not success:
        return False, msg

    return True, msg