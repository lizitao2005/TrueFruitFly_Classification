# TrueFruitFly_Classification
# A Multi Angle Image Dataset for Deep Learning-Based Classification of 26 Quarantine True Fruit Flies (Diptera: Tephritidae)
Scripts to train all the models in the paper for true quarantine fruit flies identification.

All models were trained on CAU_HPC hosted by China Agricultral University.

See this publication for more information on model training and the dataset: link to paper will be added when it is published.

Dataset 1-3 and Tephritid26 are available here: link to figshare repo will be added when it is published.

Usage:
Training:
e.g. python Train.py --lr 0.0005 --output_dir "directory_of_results" --data_dir "directory_of_dataset" --batch_size 32 --num_epochs 20 --model_type resnet

Predicting:
e.g. python Predict.py --model_path "directory_of_model(.pth)" --test_root "directory_of_test" --model_type resnet --results_dir "directory_of_results"

Grad-CAM Analysis:
e.g. python Grad-CAM.py --test_dir "directory_of_test" --output_dir "directory_of_results" --model_path "directory_of_models(.pth)"

Trainging history, confusion matrix, classification report will be generated after running scripts.
