# TrueFruitFly_Classification
# A Multi Angle Image Dataset for Deep Learning-Based Classification of 26 Quarantine True Fruit Flies (Diptera: Tephritidae)
Scripts to train all the models in the paper for true quarantine fruit flies identification.

The model was trained on CAU_HPC hosted by China Agricultral University

See this publication for more information on model training and the dataset: link to paper will be added when it is published.
The dataset and the model weights are available here: link to figshare repo will be added when it is published.


Usage:
e.g. python train.py --lr 0.0005 --output_dir Batch32_lr00005_epoch20 --data_dir data_dir --batch_size 32 --num_epochs 20 --model_type resnet
e.g. python Predict.py --model_path model_path --test_root test_root --model_type resnet --results_dir results_dir
