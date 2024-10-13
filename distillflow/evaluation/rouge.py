import evaluate

def read_file(file_path):
    """Reads text from a file and returns it as a list of lines."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]


def compute_rouge_scores(reference_file, generated_file):
    # Load the ROUGE metric from the evaluate library
    rouge = evaluate.load('rouge')

    reference_texts = read_file(reference_file)
    generated_texts = read_file(generated_file)

    # Compute ROUGE scores for the provided texts
    scores = rouge.compute(predictions=generated_texts, references=reference_texts)

    # Return the scores
    return scores


# Example usage
# reference_texts = ["The cat sat on the mat."]
# generated_texts = ["The cat is sitting on the mat."]
#
# # Compute and print ROUGE scores
# rouge_scores = compute_rouge_scores(reference_texts, generated_texts)
# for metric, score in rouge_scores.items():
#     print(f"{metric}: {score:.4f}")
