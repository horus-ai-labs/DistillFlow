from transformers import AutoModelForQuestionAnswering, AutoTokenizer, BatchEncoding
import torch
from distillflow.student import Student
import difflib

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


    def find_best_token_match(self, input_ids, answer_token_ids):
        """Find the best matching span of tokens using approximate matching."""
        input_str = " ".join(map(str, input_ids))
        answer_str = " ".join(map(str, answer_token_ids))

        matcher = difflib.SequenceMatcher(None, input_str, answer_str)
        match = matcher.find_longest_match(0, len(input_str), 0, len(answer_str))

        if match.size > 0:
            start = match.a // len(answer_str.split())
            return start, start + len(answer_token_ids) - 1
        return None, None


    def encode(self, batch) -> BatchEncoding:
        questions = batch['prompt']
        contexts = batch['context']
        answers = batch['response']

        # Tokenize inputs (context and questions)
        inputs = self.tokenizer(
            questions,
            contexts,
            max_length=512,
            truncation=True,
            # stride=128,
            # return_overflowing_tokens=True,
            # return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )

        print(f"batch size: {len(contexts)}: {len(questions)}: {len(answers)}: {len(inputs)}")

        # Initialize start and end positions with the same batch size as inputs
        batch_size = len(questions)
        start_positions = torch.zeros(batch_size, dtype=torch.long)
        end_positions = torch.zeros(batch_size, dtype=torch.long)

        for i, answer_text in enumerate(answers):
            input_ids = inputs["input_ids"][i]

            if len(input_ids) > 512:
                input_ids = input_ids[:512]
                inputs["input_ids"][i] = input_ids

            # Tokenize the answer to find its position in the context
            answer_token_ids = self.tokenizer(answer_text)["input_ids"]

            # Find start position of the answer in the tokenized context
            start_pos, end_pos = self.find_best_token_match(input_ids, answer_token_ids)
            if start_pos is None or end_pos is None:
                # try:
                #     start_pos = inputs["input_ids"][i].tolist().index(answer_token_ids[1])
                #     end_pos = start_pos + len(answer_token_ids) - 2
                # except ValueError as e:
                #     print(e)
                    # print(f"ValueError for answer_text: {answer_text}, {contexts[i]}, {questions[i]}")
                    # Handle cases where the answer is not found in the context
                start_pos = 0
                end_pos = 0

                    # print(f"start and end not found: {answer_text}: {valid_contexts[i]}\n\n")
                    # start_pos, end_pos = 0, 0

            start_positions[i] = start_pos
            end_positions[i] = end_pos

        inputs["start_positions"] = torch.tensor(start_positions)
        inputs["end_positions"] = torch.tensor(end_positions)

        print(inputs['input_ids'].shape)
        print(inputs['start_positions'].shape)
        return inputs

    def forward_pass(self, batch):
        return self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            start_positions=batch['start_positions'],
            end_positions=batch['end_positions']
        )