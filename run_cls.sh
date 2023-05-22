# set exp_name and class_choice
exp_name=cls_dgcnn
model=xdgcnn_dgcnn
dropout=0.1
echo "exp_name: $exp_name"
echo "model: $model"

python main_cls.py --exp_name=${exp_name}_1024 --num_points=1024 --k=20 --model=${model} --dropout=${dropout}
python main_cls.py --exp_name=${exp_name}_1024_eval --num_points=1024 --k=20 --eval=True --model_path=outputs/${exp_name}_1024/models/model.t7 --model=${model}
python main_cls.py --exp_name=${exp_name}_2048 --num_points=2048 --k=40 --batch_size=32 --model=${model} --dropout=${dropout}
python main_cls.py --exp_name=${exp_name}_2048_eval --num_points=2048 --k=40 --batch_size=32 --eval=True --model_path=outputs/${exp_name}_2048/models/model.t7 --model=${model}
python main_cls.py --exp_name=${exp_name}_2048_b16 --num_points=2048 --k=40 --batch_size=16 --model=${model} --dropout=${dropout}
python main_cls.py --exp_name=${exp_name}_2048_b16_eval --num_points=2048 --k=40 --batch_size=16 --eval=True --model_path=outputs/${exp_name}_2048_b16/models/model.t7 --model=${model}
