import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
# from trl import SFTTrainer
from accelerate import Accelerator

from distillflow.distill.distiller import Distiller
from distillflow.student.distillbert import DistillBert


from trl import SFTTrainer
from transformers import TrainingArguments
class SFT(Distiller):
    def __init__(self, student_model=DistillBert()):
        """
        Initialize the student model.
        Args:
            model_name: Name of the pre-trained student model.
            device: Device to run the model on, defaults to GPU if available.
        """
        super().__init__(student_model)
        # self.accelerator = Accelerator()

    def fine_tune(self, dataset, output_dir='./outputs', epochs=3, learning_rate=1e-4):
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

        dataset = dataset.map(self.student.encode, batched=True)

        print(dataset)

        trainer = SFTTrainer(
            model=self.student.model,
            tokenizer=self.student.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length= 2048,
            dataset_num_proc=2,
            packing=False,  # Can make training 5x faster for short sequences.
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                # num_train_epochs = 1, # Set this for 1 full training run.
                max_steps=60,
                learning_rate=2e-4,
                # fp16=True,
                # bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_hf",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
            ),
        )

        trainer_stats = trainer.train()

        #Save models
        self.student.save_pretrained(output_dir)
        self.student.tokenizer.save_pretrained(output_dir)


        # train_dataloader = self.prepare_dataloader(dataset)
        #
        # # Optimizer and scheduler setup
        # optimizer = torch.optim.AdamW(self.student.model.parameters(), lr=learning_rate)
        # # lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=epochs * len(train_dataloader))
        #
        # # trainer = SFTTrainer(
        # #     "facebook/opt-350m",
        # #     train_dataset=dataset,
        # #     args=sft_config,
        # # )
        # # Move model and optimizer to the appropriate devices
        # self.student.model, optimizer, train_dataloader = self.accelerator.prepare(
        #     self.student.model, optimizer, train_dataloader
        # )
        #
        # # Training loop
        # for epoch in range(epochs):
        #     self.student.model.train()
        #     for step, batch in enumerate(train_dataloader):
        #         batch = {k: torch.tensor(v) if isinstance(v, list) else v for k, v in batch.items()}
        #
        #         # Move batch to the correct device (accelerator handles this)
        #         batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
        #
        #         # Forward pass
        #         outputs = self.student.forward_pass(batch)
        #         loss = outputs.loss
        #
        #         # Backward pass
        #         self.accelerator.backward(loss)
        #
        #         # Optimizer step and zero gradients
        #         optimizer.step()
        #         optimizer.zero_grad()
        #         if step % 10 == 0:
        #             print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item()}")
        #
        # # Save the fine-tuned model
        # self.accelerator.wait_for_everyone()
        # unwrapped_model = self.accelerator.unwrap_model(self.student.model)
        # unwrapped_model.save_pretrained(output_dir)
        # self.student.tokenizer.save_pretrained(output_dir)
        # print(f"Model fine-tuned and saved to {output_dir}")