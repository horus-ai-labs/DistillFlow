.. image:: https://s3.us-west-2.amazonaws.com/www.horusailabs.com/distillflow.png
    :target: https://horusailabs.com/
========

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: LICENSE
.. image:: https://img.shields.io/badge/status-alpha-red.svg
    :target: https://github.com/username/DistillFlow

Overview
========

DistillFlow is an open-source toolkit designed to simplify and scale the distillation of large language models (LLMs) into smaller, more efficient models. It provides a flexible pipeline for distillation, fine-tuning, and experimentation across multiple GPUs, with support for dynamic resource allocation and easy integration of custom techniques.

DistillFlow is maintained by HorusAILabs_.

.. _HorusAILabs: https://www.horusailabs.com/

Architecture
============
DistillFlow lets you build a fully configurable pipeline, to help with your Distillation.
Gather your training data from S3, CSV or just scrape it with a web crawler.
Once the data is available, choose a teacher model, and distill multiple models in parallel.
Finally, compare the performance across all distilled models to choose the best one.


.. image:: https://s3.us-west-2.amazonaws.com/www.horusailabs.com/distillflow-architecture.png
    :target: https://s3.us-west-2.amazonaws.com/www.horusailabs.com/distillflow-architecture.png

Key Features
============
- **Multi-Strategy Distillation:** Supports multiple distillation techniques such as Teacher-Student, Knowledge Distillation, and Layer Dropout.
- **Dynamic Resource Allocation:** Automatically distributes tasks across GPUs or nodes based on available memory.
- **Fine-Tuning Support:** Allows for domain-specific and downstream fine-tuning of distilled models.
- **Profiling and Optimization:** Monitors GPU utilization and optimizes memory usage with gradient checkpointing and automatic mixed precision.
- **Easy Integration:** Compatible with popular libraries like Hugging Face Transformers, PyTorch, and DeepSpeed.


Requirements
============

* Python 3.9+
* Works on Linux, Windows, macOS

Install
=======

The quick way:

.. code-block:: bash

    pip install distillflow

Data
======
- Training Data available from:
- - [dolly](https://huggingface.co/datasets/MiniLLM/dolly),
- - [self-inst](https://huggingface.co/datasets/MiniLLM/self-inst),
- - [vicuna](https://huggingface.co/datasets/MiniLLM/Vicuna),
- - [sinst](https://huggingface.co/datasets/MiniLLM/sinst), and
- - [uinst](https://huggingface.co/datasets/MiniLLM/uinst)
- The plain-text corpus $\mathcal{D}_\text{PT}$ can be download from the HugginFace datasets [repository](https://huggingface.co/datasets/openwebtext). For reproducibility, we recommend you to use the following preprocessed data.
- The processed data can be downloaded from the following links: [dolly](https://huggingface.co/datasets/MiniLLM/dolly-processed), [openwebtext](https://huggingface.co/datasets/MiniLLM/openwebtext-processed), [roberta-corpus](https://huggingface.co/datasets/MiniLLM/roberta-corpus-processed).

Quick Start
===========
Here's a quick example to get started with DistillFlow:

.. code-block:: python

   from distillflow import DistillConfig, DistillPipeline

   # 1. Set up the configuration for the distillation process
   config = DistillConfig(
       teacher_model='EleutherAI/gpt-neo-2.7B',  # Choose the teacher model
       student_model='distilbert-base-uncased',   # Choose the student model
       training_data='dolly/self-instruct',       # Training dataset (default)
       prompt_template="Summarize: {input_text}", # Optional prompt template
       max_epochs=3,                              # Customize training parameters
       learning_rate=1e-4
   )

   # 2. Create the distillation pipeline with the specified configuration
   pipeline = DistillPipeline(config)

   # 3. Start the distillation process
   pipeline.distill()

   # (Optional) Test new configurations or add custom distillation workflows
   config.prompt_template = "Rewrite the following: {input_text}"
   pipeline.update_config(config)
   pipeline.distill()  # Run with updated settings

   # Optionally fine-tune the distilled model
   pipeline.fine_tune(training_data)

Documentation
=============

Check out our full documentation at: https://distillflow.readthedocs.io/

Configurable Parameters
=======================
DistillFlow allows users to specify and customize several parameters to control the distillation process:

- **`teacher_model`**: The path or name of the pretrained teacher model to distill from.
- **`student_model`**: The path or name of the student model to train.
- **`training_data`**: Location or name of the dataset to be used (default: `dolly/self-instruct`).
- **`prompt_template`**: Custom prompt template for text-based distillation tasks.
- **`max_epochs`**: Number of training epochs.
- **`learning_rate`**: Learning rate for training.

For a complete list of configuration options, refer to our `documentation <https://distillflow.readthedocs.io/en/latest/config.html>`_.

Contributing
============
We welcome contributions! Please see our `CONTRIBUTING.rst <https://github.com/username/DistillFlow/CONTRIBUTING.rst>`_ file for more details on how to get involved.

License
=======
Distributed under the MIT License. See `LICENSE <https://github.com/username/DistillFlow/LICENSE>`_ for more information.

Community and Support
=====================
- Join the discussion on our `GitHub Discussions <https://github.com/username/DistillFlow/discussions>`_.
- Report issues and request features using our `Issue Tracker <https://github.com/username/DistillFlow/issues>`_.