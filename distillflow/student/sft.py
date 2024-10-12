import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import get_linear_schedule_with_warmup
# from trl import SFTTrainer
from accelerate import Accelerator

class SFTStudent:
    """
    SFTStudent model class for supervised fine-tuning using collected data, optimized using Accelerate.
    """
    def __init__(self, model_name='distilgpt2', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the student model.
        Args:
            model_name: Name of the pre-trained student model.
            device: Device to run the model on, defaults to GPU if available.
        """
        self.model_name = model_name
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.accelerator = Accelerator()

    def prepare_dataloader(self, train_dataset, batch_size=4):
        """
        Prepares a PyTorch DataLoader for training.
        Args:
            train_dataset: The training dataset.
            batch_size: Batch size for training.
        """
        def encode(batch):
            inputs = batch["prompt"]
            targets = batch["response"]
            self.tokenizer.pad_token = self.tokenizer.eos_token
            model_inputs = self.tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(targets, max_length=512, truncation=True, padding="max_length")
            model_inputs["labels"] = labels["input_ids"]  # Model expects 'labels' for the target sequence.
            return model_inputs

        train_dataset = train_dataset.map(encode, batched=True)
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        print(f"Training Dataset: {train_dataset}")
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_dataloader

    def fine_tune(self, train_dataset, output_dir='./sft_output', epochs=3, learning_rate=1e-4):
        """
        Fine-tune the student model using the provided training dataset with Accelerate.
        Args:
            train_dataset: The training dataset.
            output_dir: Directory to save the fine-tuned model.
            epochs: Number of training epochs.
            learning_rate: Learning rate for training.
        """
        # Prepare DataLoader
        train_dataloader = self.prepare_dataloader(train_dataset)

        # Optimizer and scheduler setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        # lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=epochs * len(train_dataloader))

        # trainer = SFTTrainer(
        #     "facebook/opt-350m",
        #     train_dataset=dataset,
        #     args=sft_config,
        # )
        # Move model and optimizer to the appropriate devices
        self.model, optimizer, train_dataloader = self.accelerator.prepare(
            self.model, optimizer, train_dataloader
        )

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                # Move batch to the correct device (accelerator handles this)
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)  # This will use input_ids, attention_mask, and labels internally
                loss = outputs.loss

                # Backward pass
                self.accelerator.backward(loss)

                # Optimizer step and zero gradients
                optimizer.step()
                optimizer.zero_grad()

    # for step, batch in enumerate(train_dataloader):
            #     outputs = self.model(**batch)
            #     loss = outputs.loss
            #     self.accelerator.backward(loss)
            #     optimizer.step()
            #     # lr_scheduler.step()
            #     optimizer.zero_grad()
            #
            #     if step % 10 == 0:
            #         print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item()}")

        # Save the fine-tuned model
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model fine-tuned and saved to {output_dir}")