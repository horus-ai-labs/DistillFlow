# file to convert datasets to ShareGPT format.
import argparse
from datasets import load_dataset, DatasetDict

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='hugging face dataset path')
    parser.add_argument('--subset-name', type=str, default="all", help='dataset subset name')
    parser.add_argument('--parser-type', type=str, choices=['gsm8k', 'mmlu', 'wikisql'])
    parser.add_argument('--split', type=str, default='auxiliary_train,validation,test', help='dataset split')
    parser.add_argument('--parsed-dataset-name', type=str, required=True, help='')

    return parser.parse_args()

def format_map_mmlu(item):
    question = item.get("question", "").strip()
    choices = item.get("choices", [])
    choices = [choice.strip() for choice in choices]
    correct_choice = item.get("answer", 0)  # Correct answer key (e.g., "A", "B", etc.)

    # Construct assistant's response
    assistant_response = f"The correct answer is {correct_choice}: {choices[correct_choice]}""."

    # ShareGPT format
    conversation = {
        "conversations": [
            {"from": "human",
             "value": f"Question: {question}\n\nChoices:\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"},
            {"from": "gpt", "value": assistant_response},
        ],
    }
    return conversation

def format_map_gsm8k(item):
    question = item.get("question", "").strip()
    answer = item.get("answer", "").strip()  # Correct answer key (e.g., "A", "B", etc.)

    # ShareGPT format
    conversation = {
        "conversations": [
            {"from": "human",
             "value": f"Question: {question}"},
            {"from": "gpt", "value": answer},
        ],
    }
    return conversation


def format_wikisql(item):
    table_columns = ", ".join(item["table"]["header"])  # Extract table schema
    question = item["question"]  # Extract user question
    sql_query = item["sql"]["human_readable"]  # Extract SQL query

    conversation = {
        "conversations": [
            {
                "from": "human",
                "value": f"Considering the provided database schema and associated query, produce SQL code to retrieve the answer to the query. \n###Database Schema: {table_columns}\n###Question: {question}"
            },
            {
                "from": "gpt",
                "value": sql_query
            }
        ]
    }

    return conversation


def get_parser(parser_type):
    if parser_type == 'gsm8k':
        return format_map_gsm8k
    elif parser_type == 'mmlu':
        return format_map_mmlu
    elif parser_type == 'wikisql':
        return format_wikisql

    else:
        print("Pass a valid parser for dataset")
        exit()


# Load the dataset
def main(args):
    combined_dataset = DatasetDict()
    parser = get_parser(args.parser_type)
    for split in args.split.split(','):
    #multiple choice type of datasets.
        dataset = load_dataset(args.dataset, args.subset_name, split=split)
        dataset = dataset.map(parser, remove_columns=dataset.column_names)
        combined_dataset[split] = dataset

    combined_dataset.push_to_hub(args.parsed_dataset_name)

if __name__ == '__main__':
    main(parse_arguments())