from flask import Flask, request, jsonify
import requests
from pathlib import Path
import os
import cv2
import shutil
from ultralytics import YOLO
import yaml
import torch

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

            #  확장자 자동 추출
            extension = os.path.splitext(url.split("?")[0])[1]  # URL에서 확장자 추출 (쿼리스트링 제거)
            if not extension:  
                extension = ".jpg"  # 확장자 없으면 기본값 jpg
            
            #  확장자 반영해서 저장
            with open(label_dir / f"{i}{extension}", "wb") as f:
                f.write(img_res.content)
            downloaded_count += 1

        except Exception as e:
            print(f"Error downloading image {url}: {e}")
    if downloaded_count == 0:
        return False, "All downloads failed"

    return True, str(label_dir)

# ⑥ 모델 학습 함수 << 코드 넣어주세요!
#어떤 데이터 필요하신지 몰라서 일단 이미지 경로만 가져왔습니다
################################################################\

def fix_data_yaml(dataset_folder):
    """
    data.yaml 파일의 경로를 절대 경로로 수정
    
    Args:
        dataset_folder (str): 데이터셋 폴더 경로
    """
    dataset_path = Path(dataset_folder)
    yaml_path = dataset_path / 'data.yaml'
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml 파일을 찾을 수 없습니다: {yaml_path}")
    
    # 현재 yaml 파일 읽기
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # 절대 경로로 변경
    current_dir = os.getcwd()
    
    # 각 split의 경로를 절대 경로로 변경
    for split in ['train', 'val', 'test']:
        if split in data:
            relative_path = data[split]
            # 현재 경로를 기준으로 절대 경로 생성
            absolute_path = os.path.abspath(os.path.join(current_dir, relative_path))
            data[split] = absolute_path
            print(f"{split} 경로: {absolute_path}")
    
    # 수정된 yaml 파일 저장
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"data.yaml 파일이 수정되었습니다: {yaml_path}")
    return str(yaml_path)

def check_dataset_structure(dataset_folder):
    """
    데이터셋 구조 확인 및 누락된 폴더 생성
    
    Args:
        dataset_folder (str): 데이터셋 폴더 경로
    """
    dataset_path = Path(dataset_folder)
    
    print("데이터셋 구조 확인 중...")
    
    # 필요한 폴더 구조
    required_dirs = [
        'train/images', 'train/labels',
        'val/images', 'val/labels',
        'test/images', 'test/labels'
    ]
    
    for dir_path in required_dirs:
        full_path = dataset_path / dir_path
        if not full_path.exists():
            print(f"누락된 폴더 생성: {full_path}")
            full_path.mkdir(parents=True, exist_ok=True)
        else:
            # 폴더 내 파일 개수 확인
            if 'images' in dir_path:
                image_count = len(list(full_path.glob('*.jpg'))) + len(list(full_path.glob('*.png')))
                print(f"{dir_path}: {image_count}개 이미지")
            elif 'labels' in dir_path:
                label_count = len(list(full_path.glob('*.txt')))
                print(f"{dir_path}: {label_count}개 라벨")
    
    # data.yaml 파일 확인
    yaml_path = dataset_path / 'data.yaml'
    if yaml_path.exists():
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        print(f"클래스 수: {data.get('nc', 'N/A')}")
        print(f"클래스 이름: {data.get('names', 'N/A')}")
    else:
        print("data.yaml 파일이 없습니다!")
    
    return True

def create_minimal_dataset_if_missing(dataset_folder):
    """
    데이터셋이 없는 경우 최소한의 구조 생성
    """
    dataset_path = Path(dataset_folder)
    
    if not dataset_path.exists():
        print(f"데이터셋 폴더가 없습니다. 생성합니다: {dataset_path}")
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # 기본 폴더 구조 생성
        for split in ['train', 'val', 'test']:
            (dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # 기본 data.yaml 생성
        default_yaml = {
            'train': str(dataset_path / 'train' / 'images'),
            'val': str(dataset_path / 'val' / 'images'),
            'test': str(dataset_path / 'test' / 'images'),
            'nc': 2,
            'names': ['target1', 'target2']
        }
        
        with open(dataset_path / 'data.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(default_yaml, f, default_flow_style=False)
        
        print("기본 데이터셋 구조가 생성되었습니다.")
        print("실제 이미지와 라벨 파일을 해당 폴더에 넣어주세요.")

def fixed_train():
    """수정된 학습 함수"""
    
    # GPU 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"사용 디바이스: {device}")
    
    # 데이터셋 경로
    dataset_folder = "generated_animal_dataset"
    
    try:
        # 1. 데이터셋 구조 확인
        if not os.path.exists(dataset_folder):
            print(f"데이터셋 폴더를 찾을 수 없습니다: {dataset_folder}")
            print("먼저 데이터셋을 생성해주세요.")
            create_minimal_dataset_if_missing(dataset_folder)
            return
        
        # 2. 데이터셋 구조 확인
        check_dataset_structure(dataset_folder)
        
        # 3. data.yaml 경로 수정
        yaml_path = fix_data_yaml(dataset_folder)
        
        # 4. 모델 로드
        model = YOLO('yolov8s.pt')
        
        # 5. 학습 시작
        print("\n" + "="*50)
        print("학습 시작...")
        print("="*50)
        
        results = model.train(
            data=yaml_path,
            epochs=100,
            imgsz=640,
            batch=16,
            device=device,
            project='animal_detection',
            name='train',
            exist_ok=True,
            patience=50,
            save_period=10,
            plots=True,
            val=True,
            verbose=True
        )
        
        print("\n" + "="*50)
        print("학습 완료!")
        print("="*50)
        print("최고 모델: animal_detection/train/weights/best.pt")
        print("최종 모델: animal_detection/train/weights/last.pt")
        print("학습 결과: animal_detection/train/")
        
        # 6. 검증
        print("\n검증 시작...")
        best_model = YOLO('animal_detection/train/weights/best.pt')
        val_results = best_model.val(data=yaml_path)
        
        print("검증 완료!")
        return results
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        print("\n문제 해결 방법:")
        print("1. 데이터셋 폴더가 올바른 위치에 있는지 확인")
        print("2. 이미지와 라벨 파일이 올바른 폴더에 있는지 확인")
        print("3. data.yaml 파일 내용 확인")
        raise

def debug_dataset_paths():
    """데이터셋 경로 디버깅"""
    dataset_folder = "generated_animal_dataset"
    dataset_path = Path(dataset_folder)
    
    print("=" * 50)
    print("데이터셋 경로 디버깅")
    print("=" * 50)
    
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    print(f"데이터셋 폴더 존재: {dataset_path.exists()}")
    
    if dataset_path.exists():
        print(f"데이터셋 절대 경로: {dataset_path.absolute()}")
        
        # 각 폴더 확인
        for split in ['train', 'val', 'test']:
            images_path = dataset_path / split / 'images'
            labels_path = dataset_path / split / 'labels'
            
            print(f"\n{split.upper()} 세트:")
            print(f"  이미지 폴더: {images_path.exists()} - {images_path.absolute()}")
            print(f"  라벨 폴더: {labels_path.exists()} - {labels_path.absolute()}")
            
            if images_path.exists():
                image_files = list(images_path.glob('*'))
                print(f"  이미지 파일 수: {len(image_files)}")
                if image_files:
                    print(f"  첫 번째 파일: {image_files[0].name}")
        
        # data.yaml 확인
        yaml_path = dataset_path / 'data.yaml'
        if yaml_path.exists():
            print(f"\ndata.yaml 존재: {yaml_path.absolute()}")
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            print("data.yaml 내용:")
            for key, value in data.items():
                print(f"  {key}: {value}")
        else:
            print("\ndata.yaml 파일이 없습니다!")
    
    print("=" * 50)

def manual_fix_yaml():
    """수동으로 data.yaml 파일 수정"""
    dataset_folder = "generated_animal_dataset"
    dataset_path = Path(dataset_folder)
    yaml_path = dataset_path / 'data.yaml'
    
    # 현재 디렉토리 기준 절대 경로
    current_dir = Path.cwd()
    
    # 새로운 data.yaml 생성
    new_data = {
        'train': str(current_dir / dataset_folder / 'train' / 'images'),
        'val': str(current_dir / dataset_folder / 'val' / 'images'),
        'test': str(current_dir / dataset_folder / 'test' / 'images'),
        'nc': 2,  # 클래스 수 (target1, target2)
        'names': ['target1', 'target2']  # 클래스 이름
    }
    
    # 실제 클래스 정보가 있다면 기존 파일에서 가져오기
    if yaml_path.exists():
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                existing_data = yaml.safe_load(f)
            if 'nc' in existing_data:
                new_data['nc'] = existing_data['nc']
            if 'names' in existing_data:
                new_data['names'] = existing_data['names']
        except:
            pass
    
    # 새 파일 저장
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(new_data, f, default_flow_style=False)
    
    print(f"data.yaml 파일이 수정되었습니다: {yaml_path}")
    print("새로운 경로:")
    for key, value in new_data.items():
        print(f"  {key}: {value}")


class AnimalDatasetGenerator:
    def __init__(self, model_path, main_folder, output_folder="generated_dataset"):
        """
        동물 감지 기반 자동 데이터셋 생성기
        
        Args:
            model_path (str): 학습된 animal 모델 경로 (예: 'animal.pt')
            main_folder (str): target1, target2 등의 폴더가 있는 메인 폴더 경로
            output_folder (str): 생성될 데이터셋 폴더 이름
        """
        self.model = YOLO(model_path)
        self.main_folder = Path(main_folder)
        self.output_folder = Path(output_folder)
        
        # 지원하는 이미지 및 비디오 확장자
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        
        # 데이터셋 폴더 구조 생성
        self.setup_dataset_structure()
        
        # 클래스 정보 저장
        self.class_names = []
        self.class_count = 0
        
    def setup_dataset_structure(self):
        """YOLO 형식의 데이터셋 폴더 구조 생성"""
        # 메인 데이터셋 폴더 생성
        self.output_folder.mkdir(exist_ok=True)
        
        # train, val, test 폴더 생성
        for split in ['train', 'val', 'test']:
            (self.output_folder / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_folder / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def get_target_folders(self):
        """메인 폴더에서 target로 시작하는 폴더들 찾기"""
        target_folders = []
        for folder in self.main_folder.iterdir():
            if folder.is_dir() and folder.name.startswith('target'):
                target_folders.append(folder)
        return target_folders
    
    def extract_frames_from_video(self, video_path, output_dir, frame_interval=30):
        """비디오에서 프레임 추출"""
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        saved_frame_count = 0
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 지정된 간격마다 프레임 저장
            if frame_count % frame_interval == 0:
                frame_filename = f"{video_path.stem}_frame_{saved_frame_count:04d}.jpg"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                frames.append(frame_path)
                saved_frame_count += 1
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def detect_animals_in_image(self, image_path, confidence_threshold=0.25):
        """이미지에서 동물 감지"""
        results = self.model.predict(source=str(image_path), conf=confidence_threshold, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return []
        
        detections = []
        for box in results[0].boxes:
            # YOLO 형식으로 좌표 변환 (normalized coordinates)
            x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2 / results[0].orig_shape[1]
            y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2 / results[0].orig_shape[0]
            width = (box.xyxy[0][2] - box.xyxy[0][0]) / results[0].orig_shape[1]
            height = (box.xyxy[0][3] - box.xyxy[0][1]) / results[0].orig_shape[0]
            
            detections.append({
                'x_center': float(x_center),
                'y_center': float(y_center),
                'width': float(width),
                'height': float(height),
                'confidence': float(box.conf[0])
            })
        
        return detections
    
    def save_yolo_annotation(self, detections, class_id, annotation_path):
        """YOLO 형식의 어노테이션 파일 저장"""
        with open(annotation_path, 'w') as f:
            for detection in detections:
                line = f"{class_id} {detection['x_center']:.6f} {detection['y_center']:.6f} {detection['width']:.6f} {detection['height']:.6f}\n"
                f.write(line)
    
    def process_target_folder(self, target_folder, class_id, split='train'):
        """target 폴더 내의 모든 이미지와 비디오 처리"""
        processed_count = 0
        
        # 임시 폴더 생성 (비디오 프레임 추출용)
        temp_frames_dir = self.output_folder / 'temp_frames'
        temp_frames_dir.mkdir(exist_ok=True)
        
        for file_path in target_folder.rglob('*'):
            if not file_path.is_file():
                continue
                
            file_extension = file_path.suffix.lower()
            
            # 이미지 파일 처리
            if file_extension in self.image_extensions:
                detections = self.detect_animals_in_image(file_path)
                
                if detections:  # 동물이 감지된 경우만 처리
                    # 이미지 복사
                    new_image_name = f"{target_folder.name}_{file_path.stem}_{processed_count:04d}.jpg"
                    new_image_path = self.output_folder / split / 'images' / new_image_name
                    shutil.copy2(file_path, new_image_path)
                    
                    # 어노테이션 파일 생성
                    annotation_path = self.output_folder / split / 'labels' / f"{new_image_name.replace('.jpg', '.txt')}"
                    self.save_yolo_annotation(detections, class_id, annotation_path)
                    
                    processed_count += 1
                    print(f"처리됨: {file_path.name} -> {new_image_name} (동물 {len(detections)}개 감지)")
            
            # 비디오 파일 처리
            elif file_extension in self.video_extensions:
                print(f"비디오 처리 중: {file_path.name}")
                
                # 비디오에서 프레임 추출
                frames = self.extract_frames_from_video(file_path, temp_frames_dir)
                
                for frame_path in frames:
                    detections = self.detect_animals_in_image(frame_path)
                    
                    if detections:  # 동물이 감지된 경우만 처리
                        # 이미지 복사
                        new_image_name = f"{target_folder.name}_{frame_path.stem}_{processed_count:04d}.jpg"
                        new_image_path = self.output_folder / split / 'images' / new_image_name
                        shutil.copy2(frame_path, new_image_path)
                        
                        # 어노테이션 파일 생성
                        annotation_path = self.output_folder / split / 'labels' / f"{new_image_name.replace('.jpg', '.txt')}"
                        self.save_yolo_annotation(detections, class_id, annotation_path)
                        
                        processed_count += 1
                        print(f"처리됨: {frame_path.name} -> {new_image_name} (동물 {len(detections)}개 감지)")
                
                # 임시 프레임 파일들 삭제
                for frame_path in frames:
                    frame_path.unlink()
        
        # 임시 폴더 삭제
        if temp_frames_dir.exists():
            shutil.rmtree(temp_frames_dir)
        
        return processed_count
    
    def create_data_yaml(self):
        """YOLO 학습용 data.yaml 파일 생성"""
        data_yaml = {
            'train': str(self.output_folder / 'train' / 'images'),
            'val': str(self.output_folder / 'val' / 'images'),
            'test': str(self.output_folder / 'test' / 'images'),
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        yaml_path = self.output_folder / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"data.yaml 파일이 생성되었습니다: {yaml_path}")
    
    def generate_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """전체 데이터셋 생성 프로세스"""
        print("데이터셋 생성을 시작합니다...")
        
        # target 폴더들 찾기
        target_folders = self.get_target_folders()
        
        if not target_folders:
            print("target으로 시작하는 폴더를 찾을 수 없습니다.")
            return
        
        print(f"발견된 target 폴더: {[folder.name for folder in target_folders]}")
        
        # 클래스 이름 설정
        self.class_names = [folder.name for folder in target_folders]
        self.class_count = len(self.class_names)
        
        total_processed = 0
        
        # 각 target 폴더 처리
        for class_id, target_folder in enumerate(target_folders):
            print(f"\n{'='*50}")
            print(f"처리 중: {target_folder.name} (클래스 ID: {class_id})")
            print(f"{'='*50}")
            
            # 임시로 train split에 모든 데이터 저장
            processed_count = self.process_target_folder(target_folder, class_id, 'train')
            total_processed += processed_count
            
            print(f"{target_folder.name} 폴더에서 {processed_count}개 이미지 처리 완료")
        
        # 데이터 분할 (train/val/test)
        self.split_dataset(train_ratio, val_ratio, test_ratio)
        
        # data.yaml 파일 생성
        self.create_data_yaml()
        
        print(f"\n{'='*50}")
        print("데이터셋 생성 완료!")
        print(f"총 처리된 이미지: {total_processed}개")
        print(f"클래스 수: {self.class_count}개")
        print(f"클래스 이름: {self.class_names}")
        print(f"데이터셋 위치: {self.output_folder}")
        print(f"{'='*50}")
    
    def split_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """데이터셋을 train/val/test로 분할"""
        print("\n데이터셋 분할 중...")
        
        # train 폴더의 모든 이미지 파일 가져오기
        train_images = list((self.output_folder / 'train' / 'images').glob('*.jpg'))
        train_images.sort()
        
        total_images = len(train_images)
        
        # 분할 인덱스 계산
        train_end = int(total_images * train_ratio)
        val_end = train_end + int(total_images * val_ratio)
        
        # val 데이터 이동
        for i in range(train_end, val_end):
            if i < len(train_images):
                image_path = train_images[i]
                label_path = self.output_folder / 'train' / 'labels' / f"{image_path.stem}.txt"
                
                # 이미지 이동
                new_image_path = self.output_folder / 'val' / 'images' / image_path.name
                shutil.move(str(image_path), str(new_image_path))
                
                # 라벨 이동
                if label_path.exists():
                    new_label_path = self.output_folder / 'val' / 'labels' / label_path.name
                    shutil.move(str(label_path), str(new_label_path))
        
        # test 데이터 이동
        for i in range(val_end, total_images):
            if i < len(train_images):
                image_path = train_images[i]
                label_path = self.output_folder / 'train' / 'labels' / f"{image_path.stem}.txt"
                
                # 이미지 이동
                new_image_path = self.output_folder / 'test' / 'images' / image_path.name
                shutil.move(str(image_path), str(new_image_path))
                
                # 라벨 이동
                if label_path.exists():
                    new_label_path = self.output_folder / 'test' / 'labels' / label_path.name
                    shutil.move(str(label_path), str(new_label_path))
        
        # 분할 결과 출력
        train_count = len(list((self.output_folder / 'train' / 'images').glob('*.jpg')))
        val_count = len(list((self.output_folder / 'val' / 'images').glob('*.jpg')))
        test_count = len(list((self.output_folder / 'test' / 'images').glob('*.jpg')))
        
        print(f"Train: {train_count}개, Val: {val_count}개, Test: {test_count}개")

def train_model(data_path):
    # 임시 테스트 코드입니다
    print(f"📦 Training model with data in {data_path}")
    # 사용 예제
    model_path = "animal.pt"  # 학습된 animal 모델 경로
    main_folder = "./train_data"  # target1, target2 폴더가 있는 메인 폴더
    output_folder = "generated_animal_dataset"  # 생성될 데이터셋 폴더
    
    # 데이터셋 생성기 초기화
    generator = AnimalDatasetGenerator(model_path, main_folder, output_folder)
    
    # 데이터셋 생성 (train:80%, val:10%, test:10%)
    generator.generate_dataset(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    print("데이터셋 문제 해결 및 학습 시작")
    
    # 1. 데이터셋 경로 디버깅
    debug_dataset_paths()
    
    # 2. data.yaml 수동 수정
    manual_fix_yaml()
    
    # 3. 수정된 학습 실행
    fixed_train()

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