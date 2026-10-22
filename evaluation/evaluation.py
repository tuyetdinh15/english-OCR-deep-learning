from sklearn.metrics import classification_report


def evaluate_predictions(model, generator, mapping_inverse):
    predictions = model.predict(generator, verbose=0)
    predicted_labels = [mapping_inverse[int(prediction.argmax())] for prediction in predictions]
    actual_labels = [mapping_inverse[int(label)] for label in generator.classes]
    return classification_report(actual_labels, predicted_labels, output_dict=True)


def compare_results(results):
    import pandas as pd

    return pd.DataFrame(results)
