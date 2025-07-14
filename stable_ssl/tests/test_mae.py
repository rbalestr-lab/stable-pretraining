def test_probing():
    import lightning as pl
    import torch
    import torchmetrics

    import stable_ssl as ossl
    from stable_ssl.data import transforms

    # without transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToImage(mean=mean, std=std),
    )
    train_dataset = ossl.data.HFDataset(
        path="frgfm/imagenette",
        name="160px",
        split="train",
        transform=train_transform,
    )
    train = torch.utils.data.DataLoader(
        dataset=train_dataset,
        sampler=ossl.data.sampler.RepeatedRandomSampler(train_dataset, n_views=2),
        batch_size=64,
        num_workers=20,
        drop_last=True,
    )
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToImage(mean=mean, std=std),
    )
    val = torch.utils.data.DataLoader(
        dataset=ossl.data.HFDataset(
            path="frgfm/imagenette",
            name="160px",
            split="validation",
            transform=val_transform,
        ),
        batch_size=128,
        num_workers=10,
    )
    data = ossl.data.DataModule(train=train, val=val)

    def forward(self, batch, stage):
        latent, pred, mask = self.backbone(batch["image"])
        batch["embedding"] = latent[:, 0]  # CLS token only
        if self.training:
            loss = ossl.losses.mae(self.backbone.patchify(batch["image"]), pred, mask)
            batch["loss"] = loss
        return batch

    backbone = ossl.backbone.mae.vit_base_patch16_dec512d8b()
    module = ossl.Module(backbone=backbone, forward=forward)
    linear_probe = ossl.callbacks.OnlineProbe(
        "linear_probe",
        module,
        "embedding",
        "label",
        probe=torch.nn.Linear(768, 10),
        loss_fn=torch.nn.CrossEntropyLoss(),
        metrics={
            "top1": torchmetrics.classification.MulticlassAccuracy(10),
            "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
        },
    )
    trainer = pl.Trainer(
        max_epochs=6,
        num_sanity_val_steps=1,
        callbacks=[linear_probe],
        precision="16-mixed",
        logger=False,
        enable_checkpointing=False,
    )
    manager = ossl.Manager(trainer=trainer, module=module, data=data)
    manager()


if __name__ == "__main__":
    test_probing()
