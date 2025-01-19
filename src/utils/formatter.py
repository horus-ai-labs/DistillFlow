# file to convert datasets to ShareGPT format.
import argparse
from datasets import load_dataset, DatasetDict

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='hugging face dataset path')
    parser.add_argument('--subset-name', type=str, default="all", help='dataset subset name')
    parser.add_argument('--split', type=str, default='auxiliary_train,validation,test', help='dataset split')
    parser.add_argument('--parsed-dataset-name', type=str, required=True, help='')

    return parser.parse_args()

def format_map(item):
    question = item.get("question", "").strip()
    choices = item.get("choices", [])
    choices = [choice.strip() for choice in choices]
    correct_choice = item.get("answer", 0)  # Correct answer key (e.g., "A", "B", etc.)

    # Construct assistant's response
    assistant_response = f"The correct answer is {correct_choice}: {choices[correct_choice]}""."

    # ShareGPT format
    conversation = {
        "conversation": [
            {"from": "human",
             "value": f"Question: {question}\n\nChoices:\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"},
            {"from": "gpt", "value": assistant_response},
        ],
    }
    return conversation

# Load the dataset
def main(args):
    combined_dataset = DatasetDict()
    for split in args.split.split(','):
    #multiple choice type of datasets.
        dataset = load_dataset(args.dataset, args.subset_name, split=split)
        dataset = dataset.map(format_map, remove_columns=dataset.column_names)
        combined_dataset[split] = dataset

    combined_dataset.push_to_hub(args.parsed_dataset_name, private=True)

if __name__ == '__main__':
    main(parse_arguments())

