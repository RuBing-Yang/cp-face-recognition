python3 main.py --mode test ^
                --cartoon-encoder ./model/facenet_seresnext/model_70.pth.tar ^
               --dataset-path ./data ^
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