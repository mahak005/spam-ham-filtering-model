import os
import joblib

MODEL_PATH = os.path.join("models", "spam_model.joblib")


def main():
    if not os.path.exists(MODEL_PATH):
        print("Trained model not found.")
        print("1. Download/prepare spam.csv (see README.md).")
        print("2. Run: python train_model.py")
        return

    try:
        model = joblib.load(MODEL_PATH)
    except (OSError, ValueError) as error:
        print(f"Could not load the model: {error}")
        return

    print("=== SMS Spam Detector ===")
    print("Paste a message to classify it as SPAM or NOT SPAM.")
    print("Type 'exit' to quit.")

    while True:
        try:
            message = input("\nEnter a message: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if message.lower() == "exit":
            print("Goodbye!")
            break
        if not message:
            print("Please enter a non-empty message.")
            continue

        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        classes = list(model.classes_)
        confidence = float(probabilities[classes.index(prediction)]) * 100

        if prediction == "spam":
            print(f"Result: SPAM (model confidence: {confidence:.1f}%)")
        else:
            print(f"Result: NOT SPAM / HAM (model confidence: {confidence:.1f}%)")
        print("Note: confidence is the model's estimate, not a guarantee.")


if __name__ == "__main__":
    main()
