**Go to yolov5 directory**

cd yolov5



**Install dependencies from yolov5/requirements.txt**

pip install -qr requirements.txt comet\_ml  # install

pip install torch torchvision torchaudio



**Pass arguments into yolov5/train.py for training**

python train.py --img 640 --batch 8 --epochs 10 --data coco128.yaml --weights yolov5s.pt



