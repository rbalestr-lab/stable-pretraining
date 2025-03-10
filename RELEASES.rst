
Version 0.1
-----------

- Base trainer offering the basic functionalities of stable-SSL (logging, checkpointing, data loading etc).
- Template trainers for supervised and self-supervised learning (general joint embedding, JEPA, and teacher student models).
- Examples of self-supervised learning methods : SimCLR, Barlow Twins, VicReg, DINO, MoCo, SimSiam.
- Classes to load templates neural networks (backbone, projector, etc).
- LARS optimizer.
- Linear warmup schedulers.
- Loss functions: NTXEnt, Barlow Twins, Negative Cosine Similarity, VICReg.
- Base classes for multi-view dataloaders.
- Functionalities to read the loggings and easily export the results.
- RankMe, LiDAR metrics to monitor training.
- Examples of extracting run data from WandB and utilizing it to create figures.
