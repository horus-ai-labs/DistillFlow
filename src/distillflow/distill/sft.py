import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from datasets import Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
# from trl import SFTTrainer
from accelerate import Accelerator

from distillflow.distill.distiller import Distiller
from distillflow.student import Student

class SFTWithoutKD(Distiller):
    def __init__(self, student_model: Student):
        """
        Initialize the student model.
        Args:
            model_name: Name of the pre-trained student model.
            device: Device to run the model on, defaults to GPU if available.
        """
        super().__init__(student_model)
        self.accelerator = Accelerator()

    def prepare_dataloader(self, train_dataset, batch_size=4):
        """
        Prepares a PyTorch DataLoader for training.
        Args:
            train_dataset: The training dataset.
            batch_size: Batch size for training.
        """

        questions = train_dataset['prompt']
        contexts = train_dataset['context']
        answers = train_dataset['response']

        # Filter out examples where context or question is empty
        valid_contexts = []
        valid_questions = []
        valid_answers = []

        for question, context, answer in zip(questions, contexts, answers):
            # Skip examples with empty context or question
            if question and context:
                valid_contexts.append(context)
                valid_questions.append(question)
                valid_answers.append(answer)

        train_dataset = Dataset.from_dict({
            "prompt": valid_questions,
            "context": valid_contexts,
            "response": valid_answers,
        })
        train_dataset = train_dataset.map(self.student.encode, batched=True)
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])

        print(f"Training Dataset: {train_dataset}")
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_dataloader

    def fine_tune(self, dataset, output_dir='./sft_output', epochs=3, learning_rate=1e-4):
        """
        Fine-tune the student model using the provided training dataset with Accelerate.
        Args:
            dataset: The training dataset.
            output_dir: Directory to save the fine-tuned model.
            epochs: Number of training epochs.
            learning_rate: Learning rate for training.
        """
        # Prepare DataLoader
        print(f"Fine-tuning student model {self.student.model_name}...")
        train_dataloader = self.prepare_dataloader(dataset)

        # Optimizer and scheduler setup
        optimizer = torch.optim.AdamW(self.student.model.parameters(), lr=learning_rate)
        # lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=epochs * len(train_dataloader))

        # trainer = SFTTrainer(
        #     "facebook/opt-350m",
        #     train_dataset=dataset,
        #     args=sft_config,
        # )
        # Move model and optimizer to the appropriate devices
        self.student.model, optimizer, train_dataloader = self.accelerator.prepare(
            self.student.model, optimizer, train_dataloader
        )

        # Training loop
        for epoch in range(epochs):
            self.student.model.train()
            for step, batch in enumerate(train_dataloader):
                batch = {k: torch.tensor(v) if isinstance(v, list) else v for k, v in batch.items()}

                # Move batch to the correct device (accelerator handles this)
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.student.forward_pass(batch)
                loss = outputs.loss

                # Backward pass
                self.accelerator.backward(loss)

                # Optimizer step and zero gradients
                optimizer.step()
                optimizer.zero_grad()
                if step % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item()}")

        # Save the fine-tuned model
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.student.model)
        unwrapped_model.save_pretrained(output_dir)
        self.student.tokenizer.save_pretrained(output_dir)
        print(f"Model fine-tuned and saved to {output_dir}")