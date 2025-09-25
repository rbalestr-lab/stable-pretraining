Example Gallery
===============

This gallery contains example configurations and scripts for training various self-supervised learning models using stable-pretraining.

Getting Started
---------------

Each example demonstrates different aspects of the framework:

- **SimCLR**: Contrastive learning with ResNet backbone on CIFAR-10
- **SLURM Templates**: Ready-to-use configurations for cluster training
- **Multi-probe**: Examples of using multiple evaluation probes during training

To run any example, use the ``spt`` CLI:

.. code-block:: bash

    spt examples/simclr_cifar10_config.yaml
    spt examples/simclr_cifar10_slurm.yaml -m  # For SLURM submission
