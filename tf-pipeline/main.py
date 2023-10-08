from _01_data_ingestion import load_data
from _02_data_validation import validate_data
from _03_data_preprocessing import preprocess_data
from _04_model_training import train_model, build_model, save_model
from _05_performance_evaluation import evaluate_model
from _06_model_prediction import predict

def main():
    # Step 1: Data Ingestion
    # data = load_data()

    # Step 2: Data Validation
    # validate_data(data)

    # Step 3: Data Preprocessing
    preprocessed_data = preprocess_data()

    # Step 4: Model Training
    # instantiate and build the model
    model = build_model()

    # train the model
    model = train_model(model, train_generator, validation_generator, epochs=1, batch_size=64)

    # Step 5: Performance Evaluation
    evaluate_model(model, preprocessed_data)

    # Step 6: Model Prediction
    prediction = predict(model, preprocessed_data)

if __name__ == "__main__":
    main()
