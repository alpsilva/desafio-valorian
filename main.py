from dataprep import process_dataset, process_predict_input_dataset
from model_training import (
    train_model, test_model,
    save_model, load_model,
    get_now_string
)
import typer

def main(operation: str, data_set_file_path: str = "",
        model_algorithm: str = "", saved_model_file_path: str = "",
        predict_input_file_path: str = "", predict_output_file_path: str = ""):
    safe = True
    match operation:
        case "train":
            if len(data_set_file_path) == 0:
                print("You have to specify the file path of the trained model.")
                safe = False

            if safe:
                X_train, y_train, X_test, y_test, X_valid, y_valid = process_dataset(data_set_file_path)

                if len(model_algorithm) == 0:
                    model_algorithm == "XGBoost"

                if len(saved_model_file_path) == 0:
                    now_str = get_now_string()
                    saved_model_file_path = f"model_{model_algorithm}_{now_str}.pkl"
                
                model = train_model(X_train, y_train, model_algorithm)
                save_model(model, saved_model_file_path)
                test_stats = test_model(model, X_test, y_test)

                print(f"model saved at /saved_models/{saved_model_file_path}")
                print("Stats for the trained model:")
                for key, item in test_stats:
                    print(f"{key}: {item}")

        case "predict":
            if len(saved_model_file_path) == 0:
                print("You have to specify the file path of the trained model.")
                safe = False

            if len(predict_input_file_path) == 0:
                print("You have to specify the file path for the prediction input file.")
                safe = False

            if safe:
                model = load_model(saved_model_file_path)

                prediction_input = process_predict_input_dataset(predict_input_file_path)
                output = model.predict(prediction_input)

                if len(predict_output_file_path) == 0:
                    now_str = get_now_string
                    predict_output_file_path = f"output_{now_str}"

                predict_output_file_path = f"./input_output/{predict_output_file_path}"
                output.save(predict_output_file_path)
                print(f"output saved at {predict_output_file_path}")

if __name__ == "__main__":
    typer.run(main)