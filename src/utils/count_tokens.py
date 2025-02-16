import argparse
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np

def parse_arguments():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Calculate P90 of token counts for a Hugging Face dataset.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-7B-Instruct",
                        help="Name of the Hugging Face model tokenizer (e.g., 'Qwen/Qwen-2').")
    parser.add_argument("--dataset_name", type=str, default="horus-ai-labs/gsm8k_sharegpt",
                        help="Name of the Hugging Face dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (default: 'train').")
    parser.add_argument("--column", type=str, default="conversations",
                        help="Column containing text in the dataset (default: 'text').")

    args = parser.parse_args()

    return args


def sharegpt_format(example, tokenizer):
    conversations = example['conversations']
    message = []
    answer = []

    if isinstance(conversations, list):
        for conversation in conversations:
            if isinstance(conversation, dict):
                if conversation.get('from') == 'human':
                    message.append({"role": "user", "content": conversation.get('value', '')})
                elif conversation.get('from') == 'gpt':
                    message.append({"role": "assistant", "content": conversation.get('value', '')})
                elif conversation.get('from') == 'system':
                    message.insert(0, {"role": "system", "content": conversation.get('value', '')})

    if not any(msg.get('role') == 'system' for msg in message):
        message.insert(0, {"role": "system", "content": "You are a helpful assistant."})

    text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    return {"text": text}

def main(args):
    model_name = args.model_name
    # Load the tokenizer (replace with the desired model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load a Hugging Face dataset (replace with your dataset name and split)
    dataset_name = args.dataset_name  # Replace with your dataset
    split = args.split           # Choose 'train', 'test', or 'validation'
    column = args.column           # Replace with the column that contains the text

    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)

    # Function to count tokens using the tokenizer
    def count_tokens(text):
        return len(text['input_ids'])

    token_counts = []
    for row in dataset:
        text = sharegpt_format(row, tokenizer)
        count = count_tokens(tokenizer(text['text']))
        token_counts.append(count)

    # Calculate statistics
    median_tokens = np.median(token_counts)
    p90_tokens = np.percentile(token_counts, 95)

    # Print the results
    print(f"Median number of tokens: {median_tokens}")
    print(f"90th percentile (P95) number of tokens: {p90_tokens}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)