def read_labels(file_path):
    """

    """
    with open(file_path, 'r') as file:
        labels = file.read().splitlines()
    return labels


def calculate_accuracy(ground_truth_labels, predicted_labels):
    correct_predictions = sum(1 for gt, pred in zip(ground_truth_labels, predicted_labels) if gt == pred)
    total_predictions = len(ground_truth_labels)
    accuracy = correct_predictions / total_predictions * 100
    return accuracy
