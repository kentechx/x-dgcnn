# set exp_name and class_choice
exp_name=semseg_s3dis_dgcnn
model=xdgcnn_dgcnn
dropout=0.1
echo "exp_name: $exp_name"
echo "model: $model"

for choice in 1 2 3 4 5 6
do
    # train
    python main_semseg_s3dis.py --exp_name=${exp_name} --test_area=${choice} --model=${model} --dropout=${dropout}
    # eval
    python main_semseg_s3dis.py --exp_name=${exp_name}_${choice}_eval --test_area=${choice} --eval=True --model_root=outputs/${exp_name}/models/ --model=${model}
done

# evaluate all
python main_semseg_s3dis.py --exp_name=${exp_name}_eval --test_area=all --eval=True --model_root=outputs/${exp_name}/models/ --model=${model}
