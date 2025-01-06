from accelerate import Accelerator
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from typing import Optional, Union
import os
from pathlib import Path


class DistillationTrainer:
    def __init__(
            self,
            accelerator: Accelerator,
            student_model,
            teacher_model,
            train_dataloader: DataLoader,
            eval_dataloader: Optional[DataLoader],
            optimizer,
            scheduler=None,
            max_seq_length: int = 4096,
            distillation_args: dict = {"temperature": 2.0, "alpha": 0.5},
            num_epochs: int = 3,
            output_dir: str = "./results",
            save_steps: int = 1000,
            logging_steps: int = 1,
            gradient_accumulation_steps: int = 1,
    ):
        self.accelerator = accelerator
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.student_model, self.optimizer, self.train_dataloader = student_model, optimizer, train_dataloader
        self.teacher_model = teacher_model

        # Prepare all training components with accelerator
        if self.accelerator is not None:
            self.student_model, self.optimizer, self.train_dataloader = accelerator.prepare(
                student_model, optimizer, train_dataloader
            )
            self.teacher_model = accelerator.prepare(teacher_model)
            if eval_dataloader is not None:
                self.eval_dataloader = accelerator.prepare(eval_dataloader)

            if scheduler is not None:
                self.scheduler = accelerator.prepare(scheduler)

        self.max_seq_length = max_seq_length
        self.distillation_args = distillation_args
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Create output directory
        if accelerator is not None and self.accelerator.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def pad_logits(self, student_logits, teacher_logits):
        student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
        if student_size != teacher_size:
            pad_size = abs(student_size - teacher_size)
            pad_tensor = torch.zeros(
                (*teacher_logits.shape[:-1], pad_size),
                dtype=teacher_logits.dtype,
                device=teacher_logits.device
            )
            return (
                torch.cat([student_logits, pad_tensor], dim=-1),
                teacher_logits
            ) if student_size < teacher_size else (
                student_logits,
                torch.cat([teacher_logits, pad_tensor], dim=-1)
            )
        return student_logits, teacher_logits

    def compute_loss(self, student_outputs, teacher_outputs, inputs):
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        original_loss = student_outputs.loss  # This might be calculated differently in SFTTrainer

        # Make sure logits and labels are on same device
        student_logits = student_logits.to(inputs['labels'].device)
        teacher_logits = teacher_logits.to(inputs['labels'].device)

        # Check shapes before padding
        print(f"Student logits shape: {student_logits.shape}")
        print(f"Teacher logits shape: {teacher_logits.shape}")

        student_logits, teacher_logits = self.pad_logits(student_logits, teacher_logits)

        temp = self.distillation_args["temperature"]
        student_logits_scaled = student_logits / temp
        teacher_logits_scaled = teacher_logits / temp

        # Calculate KL div loss only on non-padded positions
        attention_mask = inputs.get('attention_mask', None)
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits_student = student_logits_scaled.view(-1, student_logits_scaled.size(-1))[active_loss]
            active_logits_teacher = teacher_logits_scaled.view(-1, teacher_logits_scaled.size(-1))[active_loss]
            loss_kd = F.kl_div(
                F.log_softmax(active_logits_student, dim=-1),
                F.softmax(active_logits_teacher, dim=-1),
                reduction='batchmean'
            ) * (temp ** 2)
        else:
            loss_kd = F.kl_div(
                F.log_softmax(student_logits_scaled, dim=-1),
                F.softmax(teacher_logits_scaled, dim=-1),
                reduction='batchmean'
            ) * (temp ** 2)

        # Print losses for debugging
        print(f"KL loss: {loss_kd:.4f}")
        print(f"Original loss: {original_loss:.4f}")

        alpha = self.distillation_args["alpha"]
        final_loss = alpha * loss_kd + (1 - alpha) * original_loss
        print(f"Final loss: {final_loss:.4f}")

        return final_loss

    # def compute_loss(self, student_outputs, teacher_outputs, inputs):
    #     student_logits = student_outputs.logits
    #     teacher_logits = teacher_outputs.logits
    #     original_loss = student_outputs.loss
    #
    #     student_logits = student_logits.to(inputs['labels'].device)
    #     teacher_logits = teacher_logits.to(inputs['labels'].device)
    #     student_logits, teacher_logits = self.pad_logits(student_logits, teacher_logits)
    #
    #     temp = self.distillation_args["temperature"]
    #     student_logits_scaled = student_logits / temp
    #     teacher_logits_scaled = teacher_logits / temp
    #
    #     loss_kd = F.kl_div(
    #         F.log_softmax(student_logits_scaled, dim=-1),
    #         F.softmax(teacher_logits_scaled, dim=-1),
    #         reduction='batchmean'
    #     ) * (temp ** 2) / self.max_seq_length
    #
    #     alpha = self.distillation_args["alpha"]
    #     return alpha * loss_kd + (1 - alpha) * original_loss

    def save_checkpoint(self, step, epoch, loss):
        if self.accelerator.is_main_process:
            save_path = self.output_dir / f"checkpoint-{step}"
            self.accelerator.save_state(save_path)

    def train(self):
        total_start_time = time.time()
        global_step = 0

        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0

            # Only show progress bar on main process
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                disable=self.accelerator is not None and not self.accelerator.is_main_process
            )

            self.student_model.train()
            self.teacher_model.eval()

            for step, batch in enumerate(progress_bar):
                step_start_time = time.time()

                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**batch)
                student_outputs = self.student_model(**batch)
                loss = self.compute_loss(student_outputs, teacher_outputs, batch)
                loss = loss / self.gradient_accumulation_steps

                # Backward pass with accelerator
                if self.accelerator is None:
                    loss.backward()
                else:
                    self.accelerator.backward(loss)

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                # Gather loss from all processes
                if self.accelerator is not None:
                    loss = self.accelerator.gather(loss).mean()
                epoch_loss += loss.item() * self.gradient_accumulation_steps
                step_time = time.time() - step_start_time

                if self.accelerator is None or self.accelerator.is_main_process:
                    if global_step % self.logging_steps == 0:
                        progress_bar.set_postfix({
                            'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                            'step_time': f"{step_time:.2f}s",
                            'lr': f"{self.scheduler.get_last_lr()[0]:.2e}" if self.scheduler else "N/A"
                        })

                    if self.save_steps > 0 and global_step % self.save_steps == 0:
                        self.save_checkpoint(global_step, epoch, loss.item())

            if self.accelerator.is_main_process:
                epoch_time = time.time() - epoch_start_time
                avg_epoch_loss = epoch_loss / len(self.train_dataloader)
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"Average Loss: {avg_epoch_loss:.4f}")
                print(f"Epoch Time: {epoch_time:.2f}s")

            if self.eval_dataloader is not None:
                self.evaluate()

        # Save final model
        self.save_checkpoint(global_step, self.num_epochs - 1, loss.item())

        if self.accelerator.is_main_process:
            total_time = time.time() - total_start_time
            print(f"\nTraining completed in {total_time:.2f}s")

    def evaluate(self):
        self.student_model.eval()
        eval_loss = 0

        with torch.no_grad():
            for batch in tqdm(
                    self.eval_dataloader,
                    desc="Evaluating",
                    disable=not self.accelerator.is_main_process
            ):
                teacher_outputs = self.teacher_model(**batch)
                student_outputs = self.student_model(**batch)
                loss = self.compute_loss(student_outputs, teacher_outputs, batch)
                eval_loss += loss.item()

        if self.accelerator.is_main_process:
            avg_eval_loss = eval_loss / len(self.eval_dataloader)
            print(f"Evaluation Loss: {avg_eval_loss:.4f}")