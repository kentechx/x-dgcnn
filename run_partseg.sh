# set exp_name and class_choice
exp_name=partseg
model=dgcnn
echo "exp_name: $exp_name"
echo "model: $model"

# full dataset
python main_partseg.py --exp_name=${exp_name} --model=${model}
# eval full dataset
python main_partseg.py --exp_name=${exp_name}_eval --eval=True --model_path=outputs/${exp_name}/models/model.t7

for choice in airplane bag cap car chair earphone guitar knife lamp laptop motor mug pistol rocket skateboard table
do
    # train
    python main_partseg.py --exp_name=${exp_name}_${choice} --class_choice=${choice}
    # eval
    python main_partseg.py --exp_name=${exp_name}_${choice}_eval --class_choice=${choice} --eval=True --model_path=outputs/${exp_name}_${choice}/models/model.t7
done
