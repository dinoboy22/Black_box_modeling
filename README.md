# Black box modeling
_by Kyumin_

## 수행 방법

`test/apply_power_transform.py`
이 결과로 dataset/ 디렉토리에 power_transfomer.pkl 파일과 train_pt_excl_y.csv 파일이 생성

`test/train_baseline_model.py`
이 결과로 models/ 디렉토리에baseline-v*.ckpt 파일이 생성 (v*는 버전 번호)

`tensorboard --logdir=tb_logs`
이 후 브라우저에서 http://localhost:6006/ 접속하여 학습 결과 확인

`test/predict_and_submit.py`
이 결과로 dataset/ 디렉토리에 submission.csv 파일이 생성

`test/compare_submissions.py`
두 개의 submission.csv 파일을 비교
비교하기 전에 파일명을 변경해야 함 (예, submission1.csv, submission2.csv)

