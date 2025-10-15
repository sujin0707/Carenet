#!/bin/bash

echo ""
echo "=========================================="
echo "데이터 전처리 중..."
echo "=========================================="
python3 ./src/prepro_carenet.py
python3 ./src/prepro_well.py
python3 ./src/prepro_emo.py
python3 ./src/prepro_emer.py
python3 ./src/data_aug.py

echo ""
echo "데이터 전처리가 완료되었습니다."
