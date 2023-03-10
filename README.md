# desafio-valorian

For this challenge, the following was done:
1. Descriptive analysis on the dataset (analise_descritiva.ipynb)
2. analysis and optimization fo the requested models (analise_modelos.ipynb)
3. CLI (made with Typer) for quick training of the models, using the terminal (main and model_training). In this Module, each transformation, cleaning and training step that were done in the analyses were wrapped in separate functions and files for organization.

## How to run the CLI

1. make sure you are in the root of the project.

2. Install dependencies:
```bash
pip3 install -r requirements.txt
```

## Quick start
### Train a model
```bash
python3 main.py train --model-algorithm [RandomForest or XGBoost] --data-set-file-path ./dataset/coleta.txt
```

## Predict a dataset with a previously trained model
```bash
python3 main.py predict --saved-model-file-path ./input_output/saved_models/<model_name>.pkl --predict-input-file-path ./input_output/pred_input.csv
```

## Opções do CLI

| command                   | Description                                                                                                |
| ------------------------- | ---------------------------------------------------------------------------------------------------------- |
| operation                 | train or predict. Trains a new model or predict using an existing saved model.                             |
| data_set_file_path        | file_path to the training dataset.                                                                         |   
| model_algorithm           | XGBoost or RandomForest.                                                                                   |
| saved_model_file_path     | file path where the model will be saved for later use (if none is given, generates one automatically).     |
| predict_input_file_path   | file path where the input data for prediction is.                                                          |
| predict_output_file_path  | file path where the output of a prediction will be saved (if none is given, generates one automatically).  |


