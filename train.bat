python3 main.py --mode train-all ^
               --dataset-path ./data/splited ^
               --workers 0 ^
               --epochs 2000 ^
               --num-classes 16 ^
               --num-samples 1 ^
               --use-augmentation ^
               --use-pretrained ^
               --lr 1e-4 ^
               --attention ^
               --scheduler MultiStepLR ^
               --save-dir ./model/facenet_seresnext ^
               --gpu 0 
pause