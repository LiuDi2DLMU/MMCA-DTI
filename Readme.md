#  Source codes:


+ data: create data storage.
+ data_process: Data preprocessing.
  + dataset_.py: For constructing a balanced DrugBank dataset.
  + get_k_fold_E1.py: For dividing the k-fold cross-validation data in the E1 setting.
  + get_k_fold_E2_E3.py: For dividing the k-fold cross-validation data in the E2 or E3 setting.
  + preprocess.py: Used to preprocess data for feature coding.
+ frame: Architecture used to train the model.
+ hyperparameter.py: Used to set hyperparameters.
+ models.py: The model file.
+ module.py: Module realization.
+ train_k_fold.py: For training the model.

# Step-by-step running:

## 1. Data processing

Preprocess the dataset in the following format:

| target id | drug id | target seq | SMILES | label |
|:---------:|:-------:|:----------:|:------:|:-----:|

You can refer to the KIBA file in the data file.

## 2. Preprocessing data feature coding
Feature coding all data before training the model

```sh
python .\data_process\get_k_fold_E1.py --dataset dataset_name --k k_fold_num
```

where 'dataset_name' is for 'Davis', 'DrugBank' or 'KIBA', respectively; 'k_fold_num' is the k-fold num.

## 3. Train a prediction model with validation 

```sh
python .\data_process\get_k_fold_E2_E3.py --dataset dataset_name --k k_fold_num --e e_setting
```
