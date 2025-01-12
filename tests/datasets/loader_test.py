import unittest
import re
from functools import partial
from typing import Tuple

from datasets import Dataset, load_dataset

from distillflow.datasets.dataset_args import DataArgs, DatasetArgs
from distillflow.datasets.loader import get_dataset
from distillflow.datasets.template import ShareGpt, Alpaca, AlpacaArgs
from distillflow.model.args import ModelArguments
from distillflow.model.loader import load_tokenizer

class TestDataset(unittest.TestCase):

    def setUp(self):
        model_config = ModelArguments(
            model_name_or_path="gpt2"
        )
        # Set up common test variables
        self.tokenizer = load_tokenizer(model_config)["tokenizer"]
        self.tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

        self.data_args = DataArgs(
            train_datasets=[DatasetArgs(path="mahiatlinux/Reflection-Dataset-ShareGPT-v2", template=ShareGpt())],
            streaming=False,
            buffer_size=100,
            seed=42,
            text_field="text",
            test_size= 0.1
        )

    def test_load_train_and_eval_datasets(self):
        dataset_module = get_dataset(self.data_args, self.tokenizer, tokenize=False)
        self.assertIn("train_dataset", dataset_module)
        self.assertIn("eval_dataset", dataset_module)
        self.assertIsInstance(dataset_module["train_dataset"], Dataset)
        self.assertIsInstance(dataset_module["eval_dataset"], Dataset)
        self.assertEqual(len(dataset_module["train_dataset"]), 8253)
        self.assertEqual(len(dataset_module["eval_dataset"]), 918)
        for idx, (train_entry, test_entry) in enumerate(
                zip(dataset_module["train_dataset"], dataset_module["eval_dataset"])):
            self.validate_data_format(train_entry)
            self.validate_data_format(test_entry)

    def test_no_test_data_raises_error(self):
        self.data_args.eval_datasets = None
        self.data_args.test_size = 0
        with self.assertRaises(ValueError):
            get_dataset(self.data_args, self.tokenizer, tokenize=False)

    def test_tokenize_datasets(self):
        def tokenize_function(tokenizer, examples):
            return tokenizer(examples["text"], truncation=True, max_length=4096,
                                  padding="max_length", return_tensors="pt")
        tokenized_dataset_module = get_dataset(self.data_args, self.tokenizer, tokenize=True, tokenizer_function=partial(tokenize_function, self.tokenizer))
        self.assertIn("train_dataset", tokenized_dataset_module)
        self.assertIn("eval_dataset", tokenized_dataset_module)
        self.assertIn("input_ids", tokenized_dataset_module["train_dataset"].features)
        self.assertIn("input_ids", tokenized_dataset_module["eval_dataset"].features)
        self.assertEqual(8253, len(tokenized_dataset_module["train_dataset"]["input_ids"]))
        self.assertEqual(918, len(tokenized_dataset_module["eval_dataset"]["input_ids"]))

        def decode(tokenizer, examples):
            decoded_texts = tokenizer.batch_decode(examples["input_ids"], skip_special_tokens=False)
            concatenated_texts = "".join(decoded_texts)
            return {"text": concatenated_texts}

        decoded_texts = tokenized_dataset_module["train_dataset"].map(partial(decode, self.tokenizer), remove_columns=tokenized_dataset_module["train_dataset"].column_names)
        for idx, train_entry in enumerate(decoded_texts):
            self.validate_data_format(train_entry, False)


    def test_missing_tokenizer_function(self):
        with self.assertRaises(ValueError):
            get_dataset(self.data_args, self.tokenizer, tokenize=True)

    def test_streaming_mode(self):
        self.data_args.streaming = True
        dataset_module = get_dataset(self.data_args, self.tokenizer, tokenize=False)
        self.assertIn("train_dataset", dataset_module)
        self.assertIn("eval_dataset", dataset_module)

    def test_merge_dataset(self):
        self.data_args.train_datasets.append(DatasetArgs(path="databricks/databricks-dolly-15k", template=Alpaca(args=AlpacaArgs(
            prompt="instruction", query="context", response="response"))))

        dataset_module = get_dataset(self.data_args, self.tokenizer, tokenize=False)
        self.assertIn("train_dataset", dataset_module)
        self.assertIn("eval_dataset", dataset_module)
        self.assertIsInstance(dataset_module["train_dataset"], Dataset)
        self.assertIsInstance(dataset_module["eval_dataset"], Dataset)
        self.assertEqual(len(dataset_module["train_dataset"]), 21763)
        self.assertEqual(len(dataset_module["eval_dataset"]), 2419)

        for idx, (train_entry, test_entry) in enumerate(zip(dataset_module["train_dataset"], dataset_module["eval_dataset"])):
            self.validate_data_format(train_entry)
            self.validate_data_format(test_entry)

    def test_merge_dataset_split(self):
        self.data_args.train_datasets.append(DatasetArgs(path="databricks/databricks-dolly-15k", template=Alpaca(args=AlpacaArgs(
            prompt="instruction", query="context", response="response"))))
        self.data_args.text_field = None
        dataset_module = get_dataset(self.data_args, self.tokenizer, tokenize=False)

        mahiatlinux_dataset = load_dataset(
            "mahiatlinux/Reflection-Dataset-ShareGPT-v2",
            split="train"
        )
        total_size = len(dataset_module["train_dataset"]) + len(dataset_module["eval_dataset"])
        success, message = self.validate_split(dataset_module["train_dataset"], mahiatlinux_dataset, total_size)
        print(message)
        self.assertTrue(success)

    def validate_split(self, merged_dataset: Dataset, dataset: Dataset, total_size, tolerance=0.01) -> Tuple[bool, str]:
        """
        Validate that the proportion of `dataset` content exists in `merged_dataset`.

        Args:
            merged_dataset (Dataset): The combined dataset containing samples from multiple datasets.
            dataset (Dataset): The original dataset to check proportions for.

        Returns:
            Tuple[bool, str]: A boolean indicating if the validation passed and an explanation message.
        """
        # Extract and compare prompts (instruction) between datasets
        merged_prompts = {prompt["content"] for entry in merged_dataset for prompt in entry["_prompt"] if
                           prompt["role"] == "user"}
        dataset_prompts = {conv["value"] for entry in dataset for conv in entry["conversations"] if
                           conv["from"] == "human"}

        # # Count matches for Alpaca format
        # dataset_prompts = {entry["instruction"] for entry in dataset}
        # matched_prompts = {mp for mp in merged_prompts if any(dp in mp for dp in dataset_prompts)}
        # Count matches for ShareGPT
        matched_prompts = merged_prompts & dataset_prompts
        matched_count = len(matched_prompts)
        dataset_count = len(dataset) * len(merged_dataset) / total_size
        merged_count = len(merged_dataset)

        if dataset_count == 0 or merged_count == 0:
            return False, "Merged dataset or individual dataset is empty."

        # Calculate percentage of the dataset in the merged dataset
        original_percentage = dataset_count / merged_count
        actual_percentage = matched_count / merged_count

        # Validate against tolerance
        lower_bound = original_percentage - tolerance
        upper_bound = original_percentage + tolerance

        if lower_bound <= actual_percentage <= upper_bound:
            return True, f"Validation passed: {actual_percentage:.2%} is within the range ({lower_bound:.2%}, {upper_bound:.2%})."
        else:
            return False, f"Validation failed: {actual_percentage:.2%} is outside the range ({lower_bound:.2%}, {upper_bound:.2%})."


    def validate_data_format(self, data: dict, validate_end = True, roles=("system", "user", "assistant")):
        """
        Validate the format of the data text for compatibility with LLMs.

        Args:
            data (dict): A dictionary containing the 'text' key with formatted conversation data.
            roles (tuple): Valid roles that can appear in the text.

        Returns:
            bool: True if the format is valid, False otherwise.
        """
        text = data.get("text")
        if not text:
            raise ValueError(f"No key 'text', \nData: {data}")

        # Split the text into blocks using the tags
        blocks = re.findall(r"<\|im_start\|>(.*?)<\|im_end\|>", text, re.DOTALL)
        endoftext_blocks = re.findall(r"<\|im_start\|>(.*?)<\|endoftext\|>", text, re.DOTALL)
        blocks = list(set(blocks + endoftext_blocks))

        # Check if blocks are extracted
        if not blocks:
            raise ValueError(f"No data blocks found, \nData: {data}")

        all_roles = set()
        for block in blocks:
            # Extract the role and content
            lines = block.split("\n", 1)
            if len(lines) < 2:
                raise ValueError(f"Missing role or content for block {block}")

            role, content = lines[0].strip(), lines[1].strip()
            if role not in roles:
                return False  # Invalid role
            if not content and role != "assistant":  # Ensure non-empty content for roles other than assistant
                raise ValueError(f"Content not found for role {role} for block {block}")

            all_roles.add(role)

        if all_roles != set(roles):
            print(data)
            raise ValueError(f"All expected roles not found, found {all_roles}\nData: {data}")

        # Check if last block ends with "<|im_start|>assistant\n" if expected
        if validate_end and not text.strip().endswith("<|im_start|>assistant"):
            raise ValueError(f"Data doesn't end in the expected format\nData: {data}")

if __name__ == "__main__":
    unittest.main()