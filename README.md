# desafio-valorian

Para esse desafio, foram feitos:
1. Análise descritiva do dataset (analise_descritiva.ipynb)
2. Análise e otimização dos modelos requisitados (analise_modelos.ipynb)
3. CLI (feita utilizando a lib "typer") para treinamento rápido dos modelos via terminal (model_training). Nesse módulo, cada etapa detalhada nos notebooks de análise é replicada quando necessário e organizada em arquivos separados.

## Como rodar a CLI

1. Entre na pasta "model_training":
```bash
cd model_training
```

2. Instale as dependências:
```bash
pip3 install -r requirements.txt
```

3. Execute o arquivo principal:
```bash
python3 main.py [operation] --dataset_file_path <path/> []
```


## Opções do CLI

| command                   | Description                                                                                                                   |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| operation                 | train to train a new model or predict to use an existing saved model to predict the class, based on the contents of a file.   |
| data_set_file_path        | file_path to the training dataset.                                                                                            |   
| model_algorithm           | XGBoost or RandomForest.                                                                                                      |
| saved_model_file_path     | file path where the model will be saved for later use (if none is given, generates one automatically).                        |
| predict_input_file_path   | file path where the input data for prediction is.                                                                             |
| predict_output_file_path  | file path where the output of a prediction will be saved (if none is given, generates one automatically).                     |
