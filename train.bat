python3 main.py --mode train-all ^
               --dataset-path ./data/splited ^
               --workers 0 ^
               --epochs 3 ^
               --num-classes 16 ^
               --num-samples 1 ^
               --use-augmentation ^
               --use-pretrained ^
               --lr 1e-2 ^
               --attention ^
               --scheduler MultiStepLR ^
               --save-dir ./model/facenet_seresnext ^
               --gpu 0 ^
               --save-interval 3
python3 main.py --mode train-all ^
               --dataset-path ./data/splited ^
               --workers 0 ^
               --epochs 400 ^
               --num-classes 16 ^
               --num-samples 1 ^
               --use-augmentation ^
               --use-pretrained ^
               --lr 1e-4 ^
               --attention ^
               --scheduler MultiStepLR ^
               --save-dir ./model/facenet_seresnext ^
               --cartoon-encoder ./model/facenet_seresnext/model_3.pth.tar ^
               --gpu 0 ^
               --save-interval 10
pause