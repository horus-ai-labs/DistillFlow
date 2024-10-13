from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
from distillflow.student import Student


class DistillBert(Student):
    def __init__(self):
        super().__init__()
        self.model_name='distilbert-base-cased-distilled-squad'
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    # def encode(self, batch):
    #     inputs = batch["prompt"]
    #     targets = batch["response"]
    #     model_inputs = self.tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    #     with self.tokenizer.as_target_tokenizer():
    #         labels = self.tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    #     model_inputs["labels"] = labels["input_ids"]  # Model expects 'labels' for the target sequence.
    #     return model_inputs


    def encode(self, batch):
        inputs = batch["context"]
        questions = batch["prompt"]
        answers = batch["response"]

        # Filter out examples where context or question is empty
        valid_contexts = []
        valid_questions = []
        valid_answers = []

        for context, question, answer in zip(inputs, questions, answers):
            # Skip examples with empty context or question
            if context and question:
                valid_contexts.append(context)
                valid_questions.append(question)
                # Ensure answer is properly formatted, extracting the first answer if it's a list of dicts
                if isinstance(answer, list) and len(answer) > 0 and 'text' in answer[0]:
                    valid_answers.append(answer[0]["text"])
                else:
                    valid_answers.append("")  # Default if answer is not well-formed

        # Tokenize inputs (context and questions)
        inputs = self.tokenizer(
            valid_questions,
            valid_contexts,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Initialize start and end positions with the same batch size as inputs
        batch_size = len(valid_contexts)
        start_positions = torch.zeros(batch_size, dtype=torch.long)
        end_positions = torch.zeros(batch_size, dtype=torch.long)

        for i, answer_text in enumerate(valid_answers):
            if answer_text:  # Only process non-empty answers
                try:
                    # Tokenize the answer to find its position in the context
                    answer_token_ids = self.tokenizer(answer_text)["input_ids"]

                    # Find start position of the answer in the tokenized context
                    start_pos = inputs["input_ids"][i].tolist().index(answer_token_ids[1])
                    end_pos = start_pos + len(answer_token_ids) - 2  # Exclude [CLS] and [SEP] tokens

                    start_positions[i] = start_pos
                    end_positions[i] = end_pos
                except (ValueError, IndexError):
                    # Handle cases where the answer is not found in the context
                    start_positions[i] = 0
                    end_positions[i] = 0

        inputs["start_positions"] = torch.tensor(start_positions)
        inputs["end_positions"] = torch.tensor(end_positions)

        return inputs

    def forward_pass(self, batch):
        return self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            start_positions=batch['start_positions'],
            end_positions=batch['end_positions']
        )