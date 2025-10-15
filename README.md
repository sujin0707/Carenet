
# CARENET

발화 내용으로부터 심리적인 위험 신호를 감지하여 문맥에 따라 네 가지로 분류하고, 그에 따른 권장 대처 방안을 출력하는 모델입니다.

제공받은 데이터셋과 [웰니스 대화 스크립트 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=267), [감성 대화 말뭉치](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=86), [한국어 대화 데이터셋 일부](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=272)를 활용하여 [klue/roberta-large](https://huggingface.co/klue/roberta-large)를 finetuning하였습니다.

권장 대처 방안 출력에는 [llama-3-Korean-Bllossom-8B](https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)를 활용하였습니다.

### 실행 환경
```
Python 3.8.10
CUDA: 12.2
```

### GPU 환경
```
- GPU: NVIDIA A6000 (24GB)
- NVIDIA Driver Version: 535.161.07
- CUDA Version: 12.2
- OS: Ubuntu 20.04 LTS
```

## 실행 방법
```
conda create -n carenet
conda activate carenet
git clone https://github.com/sujin0707/Carenet.git
cd Carenet
pip install -r requirements.txt
```

## Training
```
python ./src/train.py
```

## Inference
```
python ./src/infer.py
```
OR
```
bash ./scripts/data_preprocess.sh
bash ./scripts/train.sh
bash ./scripts/infer.py
```


