import argparse

from config import MODEL_DIR
from dataset.data_preprocessing import create_generators, directory_to_dataframe, split_dataframe
from models.densenet121 import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    dataframe = directory_to_dataframe(args.dataset)
    training, validation, testing = split_dataframe(dataframe)
    train_generator, validation_generator, test_generator = create_generators(training, validation, testing)
    model = build_model(len(train_generator.class_indices))
    model.fit(train_generator, epochs=args.epochs, validation_data=validation_generator)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_DIR / "DenseNet121_model.keras"
    model.save(output_path)
    model.evaluate(test_generator)


if __name__ == "__main__":
    main()
