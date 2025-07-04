# Flask 웹 서버를 만들기 위한 기본 클래스 import
from flask import Flask, request, jsonify, render_template

# 우리가 만든 학습 코드 가져오기
from trainer import run_training_pipeline

# 우리가 만든 예측 코드 가져오기
from predictor import predict_image_from_url

import requests


# Flask 앱을 하나 만듦
app = Flask(__name__)

#기본 홈 화면
@app.route("/")
def home():
    return render_template("index.html")

# 학습을 위한 API
#이미지 다운로드까지만 구현했습니다 trainer.py에 모델 학습 코드 첨가해주시면 됩니다
@app.route("/api/train", methods=["POST"])
def train():
    data = request.json
    target_id = data.get("targetId")
    if target_id is None:
        return jsonify({"error": "targetId is required"}), 400

    success, msg = run_training_pipeline(target_id) 
    if not success:
        return jsonify({"error": msg}), 500

    return jsonify({"status": "training started", "message": msg})


# 추론을 위한 API
# 여기는 데이터를 어떻게 가져와야 할지 모르겠습니다.... find랑 findGPS 임시 데이터 firebase에 저장하는 로직만 구현했습니다
@app.route('/api/predict', methods=['POST'])
def predict_api():

    base_url = f"https://dongmul-default-rtdb.firebaseio.com/animalList"

    try:
        # 모델 예측 (현재는 2로 고정)
        target_id = predict_image_from_url()
        # 업데이트할 데이터 구성
        update_data = {
            "status": "find",
            "findGPS": "울산 남구 무거동"
            # 또는 "gps": {"lat": 35.538, "lng": 129.311}
        }

        # Firebase에 PATCH 요청
        res = requests.patch(f"{base_url}/target{target_id}.json", json=update_data)

        if res.status_code != 200:
            return jsonify({'error': 'Firebase 업데이트 실패', 'status_code': res.status_code}), 500

        return jsonify({'predicted_target': f'target{target_id}', 'message': '예측 및 업데이트 성공'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

