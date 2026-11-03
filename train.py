"""Train and evaluate the binary classification model."""

from src.ml_pipeline import train_and_evaluate, save_artifacts


def main():
    result = train_and_evaluate()
    save_artifacts(result)

    print("\nMODEL EVALUATION")
    print("-" * 45)
    for name, value in result["metrics"].items():
        print(f"{name.replace('_', ' ').title():<18}: {value:.4f}" if isinstance(value, float)
              else f"{name.replace('_', ' ').title():<18}: {value}")
    print("\nCLASSIFICATION REPORT")
    print(result["report"])
    print("Saved model to models/tumor_classifier.joblib")
    print("Saved reports to reports/")


if __name__ == "__main__":
    main()
