from trl import SFTTrainer, SFTConfig

class LogitsTrainer(SFTTrainer):
    def __init__(self, teacher_model, distillation_params, tokenizer_params):
        self.teacher_model = teacher_model
        self.distillation_params = distillation_params
        self.tokenizer_params = tokenizer_params

        super().__init__()

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        self.teacher_model = self.teacher_model.to(model.device)

        student_model = model.module if hasattr(model, 'module') else model
        teacher_model = self.teacher_model.module if hasattr(self.teacher_model, 'module') else self.teacher_model

        student_outputs = student_model(**inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)

        custom_loss = self.distillation_loss(student_outputs.logits, teacher_outputs.logits, inputs,
                                             student_outputs.loss)
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, student_logits, teacher_logits, inputs, original_loss):
        student_logits, teacher_logits = pad_logits(student_logits.to(self.model.device),
                                                    teacher_logits.to(self.model.device))

        student_logits_scaled = student_logits / self.distillation_params["temperature"]
        teacher_logits_scaled = teacher_logits / self.distillation_params["temperature"]

        loss_kd = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),
            F.softmax(teacher_logits_scaled, dim=-1),
            reduction='batchmean'
        ) * (self.distillation_params["temperature"] ** 2) / self.tokenizer_params["max_length"]

        return self.distillation_params["alpha"] * loss_kd + (1 - self.distillation_params["alpha"]) * original_loss



