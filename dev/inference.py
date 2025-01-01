import numpy as np
from sklearn.datasets import load_iris
from dev.model import IrisModel
from typing import List, Union, Tuple

# Variable to set where the artifacts will be stored
ARTIFACTS_PATH = "artifacts"


def predict_test_data() -> Tuple[float, str]:
    """
    Evaluate model performance on the test dataset.
    Returns accuracy and detailed results as a string.
    """
    # Load the trained model
    model = IrisModel.load(ARTIFACTS_PATH)

    # Load the Iris dataset for testing
    iris = load_iris()
    X_test = iris.data
    y_test = iris.target

    # Get predictions and probabilities
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Format detailed results
    results = "\nDetailed Test Predictions:\n"
    results += "=" * 80 + "\n"
    results += f"{'Input':<40} {'Predicted':<10} {'Actual':<10} {'Confidence':<10}\n"
    results += "-" * 80 + "\n"

    for i in range(len(predictions)):
        confidence = probabilities[i][predictions[i]]
        input_features = ", ".join([f"{x:.2f}" for x in X_test[i]])
        results += f"{input_features:<40} {predictions[i]:<10} {y_test[i]:<10} {confidence:.4f}\n"

    # Calculate accuracy
    accuracy = (predictions == y_test).sum() / len(y_test)

    results += f"\nOverall Accuracy: {accuracy:.4f}\n"
    results += f"Correct Predictions: {(predictions == y_test).sum()}/{len(y_test)}\n"

    return accuracy, results


def predict_custom_input(
    features: Union[List[float], np.ndarray]
) -> Tuple[int, float, List[float]]:
    """
    Make prediction for custom input features.

    Args:
        features: List or array of 4 features [sepal_length, sepal_width, petal_length, petal_width]

    Returns:
        Tuple containing:
        - predicted class (int)
        - confidence score (float)
        - probabilities for all classes (List[float])
    """
    # Input validation
    if len(features) != 4:
        raise ValueError("Input must contain exactly 4 features")

    # Convert input to numpy array
    if isinstance(features, list):
        features = np.array(features)
    features = features.reshape(1, -1)  # Reshape for single prediction

    # Load model
    model = IrisModel.load(ARTIFACTS_PATH)

    # Get prediction and probabilities
    prediction = model.predict(features)[0]  # Get single prediction
    probabilities = model.predict_proba(features)[
        0
    ]  # Get probabilities for all classes
    confidence = probabilities[prediction]

    return prediction, confidence, probabilities.tolist()


if __name__ == "__main__":
    # Example usage:
    print("\n1. Testing on iris dataset:")
    accuracy, test_results = predict_test_data()
    print(test_results)

    print("\n2. Testing with custom input:")
    try:
        # Example features: sepal length, sepal width, petal length, petal width
        sample_input = [5.1, 3.5, 1.4, 0.2]
        prediction, confidence, all_probs = predict_custom_input(sample_input)

        # Load iris for class names
        iris = load_iris()
        class_names = iris.target_names

        print(f"\nInput features: {sample_input}")
        print(f"Predicted class: {class_names[prediction]}")
        print(f"Confidence: {confidence:.4f}")
        print("\nProbabilities for each class:")
        for i, prob in enumerate(all_probs):
            print(f"{class_names[i]}: {prob:.4f}")

    except ValueError as e:
        print(f"Error: {e}")
