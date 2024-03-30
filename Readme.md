#  Source codes:


+ data: data storage.
  +  Provide Daivs and KIBA datasets here. Due to DrugBank's policy requirements, we are unable to provide its data.
  +  We provide a script in the data_process folder. Please follow DrugBank-Preprocessing. py, preprocess. py, and dataset_build. py to build a balanced dataset. Some file addresses in the script need to be adjusted according to the situation.
+ data_process: Data preprocessing.
  + dataset_build.py: For constructing a balanced DrugBank dataset.
  + DrugBank_Preprocessing.py : Used to extract the DrugBank dataset from an XML file.
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

## 2. Preprocessing data feature coding

Feature coding all data

```
python .\data_process\preprocess.py --dataset dataset_name
```

where 'dataset_name' is for 'Davis', 'DrugBank' or 'KIBA', respectively.

## 3. Data partitioning
Split the data into k-fold cross-validation according to E1, E2, E3 as illustrated in the paper

```
python .\data_process\get_k_fold_E1.py --dataset dataset_name --k k_fold_num
or
python .\data_process\get_k_fold_E2_E3.py --dataset dataset_name --k k_fold_num --e e_setting
```

where 'dataset_name' is for 'Davis', 'DrugBank' or 'KIBA', respectively; 'k_fold_num' is the k-fold num like 5 and 'e_setting' is for '2' or '3'.

## 3. Train a prediction model with validation 

```
python train_k_fold.py --dataset dataset_name --k k_fold_num --e e_setting --load_weight load_weight_file
```
where 'dataset_name' is for 'Davis', 'DrugBank' or 'KIBA', respectively; 'k_fold_num' is the k-fold num like 5; 'e_setting' is for '1', '2' or '3' and 'load_weight_file' is to load the weight file to continue training.

Then, the result will be saved.


The script will create a result folder, and the results of model training, validation, and testing will be saved to the folder based on the training set, forming the following directory format.

> - result
>  - Davis
>    - 0
>      - result.csv
>      - hyper.csv
>      - 1.pt

Among them, hyper.csv saves the parameters in this training,. pt file is the corresponding model weight file, and result.csv saves all training evaluation indicators.
