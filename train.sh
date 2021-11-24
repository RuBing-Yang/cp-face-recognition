python main.py --mode train-all \
               --dataset-path ./data/splited \
               --epochs 20 \
               --num-classes 16 \
               --num-samples 1 \
               --use-pretrained \
               --lr 1e-2 \
               --scheduler MultiStepLR \
               --attention \
               --gpu 0 
