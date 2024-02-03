def read_labels(file_path):
    with open(file_path, 'r') as file:
        labels = file.read().splitlines()
    return labels

def calculate_accuracy(ground_truth_labels, predicted_labels):
    correct_predictions = sum(1 for gt, pred in zip(ground_truth_labels, predicted_labels) if gt == pred)
    total_predictions = len(ground_truth_labels)
    accuracy = correct_predictions / total_predictions * 100
    return accuracy

def main():
    # Replace 'ground_truth.txt' and 'predicted_labels.txt' with your actual file paths
    ground_truth_file_path = 'answer_sent.txt'
    predicted_labels_file_path = 'answer_sen.txt'

    # Read labels from files
    ground_truth_labels = read_labels(ground_truth_file_path)
    predicted_labels = read_labels(predicted_labels_file_path)

    # Calculate accuracy
    accuracy = calculate_accuracy(ground_truth_labels, predicted_labels)

    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
