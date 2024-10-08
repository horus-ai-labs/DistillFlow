.. image:: https://s3.us-west-2.amazonaws.com/www.horusailabs.com/distillflow.png
    :target: https://horusailabs.com/
    :align: center
========

.. raw:: html

   <div align="center">
     <a href="LICENSE"><img src="https://img.shields.io/github/license/horus-ai-labs/DistillFlow"/></a>
     <a href="https://github.com/horus-ai-labs/DistillFlow/discussions"><img src="https://img.shields.io/badge/status-beta-red.svg"/></a>
     <a href="https://www.python.org/downloads/release/python-3120/"><img src="https://img.shields.io/badge/python-3.12-green.svg"/></a>
   </div>

Overview
========

DistillFlow is an open-source toolkit designed to simplify and scale the distillation of large language models (LLMs) into smaller, more efficient models. It provides a flexible pipeline for distillation, fine-tuning, and experimentation across multiple GPUs, with support for dynamic resource allocation and easy integration of custom techniques.

**What is distillation?**

Distillation is the process of transferring knowledge from large machine learning models to small models. The large model is often called the teacher model while the smaller model is called the student model.

DistillFlow is maintained by HorusAILabs_.

.. _HorusAILabs: https://www.horusailabs.com/

Architecture
============
DistillFlow lets you build a fully configurable pipeline, to help with your Distillation.
Once the data is available, choose a teacher model, and the student model and your dataset
and finally run the distillation.

.. raw:: html

   <p align="center">
     <img src="https://s3.us-west-2.amazonaws.com/www.horusailabs.com/distillflow_arch.png" height="600">
   </p>

Key Features
============
- **Multi-Strategy Distillation:** Supports multiple distillation techniques such as logits, attention and layers based distillation.
- **Dynamic Resource Allocation:** Automatically distributes tasks across GPUs or nodes based on available memory.
- **Fine-Tuning Support:** Allows for domain-specific and downstream fine-tuning of distilled models.
- **Model Loading Optimizations:** Supports optimized model loading using Unsloth, Liger Kernel, Flash Attention etc.
- **Easy Integration:** Compatible with popular libraries like Hugging Face Transformers, PyTorch, and DeepSpeed.

Requirements
============

* Python 3.12+
* Works on Linux, macOS

Install
=======

Clone the repository:

.. code-block:: bash

    git clone git@github.com:horus-ai-labs/DistillFlow.git
    cd DistillFlow
    pip3 install poetry
    poetry install

Data
======
We support any HuggingFace dataset in ShareGPT or Alpaca formats.

Quick Start
===========
Here's a quick example to get started with DistillFlow:

Create a training config, specifying your teacher model, student model,
huggingface dataset and the distillation type.

Use one of the existing test configs in `config` folder. The `local_distill.yaml`
works for Mac.

Here is a quick config to get started:

.. code-block:: yaml

    student_model:
        model_name_or_path: Qwen/Qwen2-1.5B
    teacher_model:
        model_name_or_path: Qwen/Qwen2-7B
    data:
      text_field: "text"
      train_datasets:
        - path: mlabonne/FineTome-100k
          template: sharegpt
    distill:
        type: logits
        max_seq_length: 1024
        sft_config:
            output_dir: './results'
            num_train_epochs: 3
            per_device_train_batch_size: 1
            gradient_accumulation_steps: 8
            eval_strategy: steps
            eval_steps: 100
            save_steps: 2000
            learning_rate: 2.0e-5
            weight_decay: 0.05
            warmup_ratio: 0.1
            lr_scheduler_type: 'cosine'
            max_grad_norm: 1.0
            group_by_length: False
      distillation_args:
            temperature: 2.0
            alpha: 0.5

Run the command:

.. code-block:: bash

    accelerate launch src/trainer.py --config <your_config_path>

Acknowledgement
=======
The repo structure is inspired by `LLamaFactory <https://github.com/hiyouga/LLaMA-Factory>`_.
The distillation training techniques are inspired by the works of `DistillKit <https://github.com/arcee-ai/DistillKit>`_.


License
=======
Distributed under the Apache-2.0 License. See `LICENSE <https://github.com/horus-ai-labs/DistillFlow/blob/main/LICENSE>`_ for more information.

Community and Support
=====================
- Join the discussion on our `GitHub Discussions <https://github.com/horus-ai-labs/DistillFlow/discussions>`_.
- Report issues and request features using our `Issue Tracker <https://github.com/horus-ai-labs/DistillFlow/issues>`_.