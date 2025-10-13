import pytest
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from unittest.mock import patch, Mock


class DummyText(nn.Module):
    """Returns a fixed [C, D] table regardless of input_ids."""

    def __init__(self, class_embeds):
        super().__init__()
        self.register_buffer("table", class_embeds)  # [C, D]

    def forward(self, input_ids=None, **kwargs):
        return types.SimpleNamespace(text_embeds=self.table)


class DummyVision(nn.Module):
    """Ignores pixel input; returns features = class_embeds[labels] for perfect top-1."""

    def __init__(self, class_embeds):
        super().__init__()
        self.ref = class_embeds  # [C, D] (buffer in DummyText)
        self.labels = None  # set per test call

    def forward(self, pixel_values=None, **kwargs):
        assert self.labels is not None, "Set .labels before calling"
        feats = self.ref[self.labels]  # [B, D]
        return types.SimpleNamespace(image_embeds=feats)


@pytest.mark.unit
class TestCLIPZeroShotUnit:
    """Unit tests for CLIPZeroShot callback."""

    @pytest.fixture
    def class_setup(self):
        C, D = 5, 8  # 5 classes, 8-dim embeddings
        torch.manual_seed(0)
        class_names = [f"class-{i}" for i in range(C)]
        class_embeds = F.normalize(torch.randn(C, D), dim=-1)  # fixed table
        tok = (
            Mock()
        )  # tokenizer_fn: return any tensor; callback only passes it into text tower
        tok.return_value = torch.ones(C, 4, dtype=torch.long)  # [C, L]
        text = DummyText(class_embeds)
        vision = DummyVision(text.table)
        metrics = {
            "top1": torchmetrics.classification.MulticlassAccuracy(
                num_classes=C, top_k=1, average="micro"
            )
        }
        return dict(
            C=C,
            D=D,
            class_names=class_names,
            class_embeds=class_embeds,
            tokenizer_fn=tok,
            text=text,
            vision=vision,
            metrics=metrics,
        )

    def _fake_pl_module(self, metrics_dict):
        """LightningModule stub with device & logging + metrics registry slot."""

        class M:
            device = torch.device("cpu")

            def __init__(self):
                self.callbacks_metrics = {}
                self.logged = {}

            def log_dict(self, logs, **kwargs):
                # store keys for assertions
                self.logged.update({k: type(v).__name__ for k, v in logs.items()})

        m = M()
        # mimic format_metrics_as_dict result structure
        m.callbacks_metrics["zs"] = {"_train": {}, "_val": metrics_dict}
        return m

    @patch("stable_pretraining.callbacks.clip_zero_shot.get_data_from_batch_or_outputs")
    @patch("stable_pretraining.callbacks.clip_zero_shot.format_metrics_as_dict")
    def test_logits_match_reference_cosine(self, fmt_mock, get_mock, class_setup):
        from stable_pretraining.callbacks.clip_zero_shot import CLIPZeroShot

        torch.manual_seed(0)
        B, C, D = 7, class_setup["C"], class_setup["D"]

        # Raw (unnormalized) features to force the callback’s own normalization path
        img_raw = torch.randn(B, D)
        cls_raw = class_setup["class_embeds"] * torch.tensor(
            [1.0, 3.0, 0.2, 5.0, 0.7]
        ).unsqueeze(1)  # uneven norms

        # Stub towers
        class RawVision(nn.Module):
            def __init__(self, feats):
                super().__init__()
                self.feats = feats

            def forward(self, pixel_values=None):
                return types.SimpleNamespace(image_embeds=self.feats)

        class RawText(nn.Module):
            def __init__(self, feats):
                super().__init__()
                self.feats = feats

            def forward(self, **kwargs):
                return types.SimpleNamespace(text_embeds=self.feats)

        vision, text = RawVision(img_raw), RawText(cls_raw)

        cb = CLIPZeroShot(
            name="zs",
            image_key="image",
            class_key="label",
            class_names=class_setup["class_names"],
            image_backbone=vision,
            text_backbone=text,
            tokenizer_fn=lambda _: torch.ones(C, 4, dtype=torch.long),
            metrics={
                "top1": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=C, top_k=1
                )
            },
        )
        fmt_mock.return_value = {"_train": {}, "_val": cb.metrics_config}
        pl_module = self._fake_pl_module({"top1": cb.metrics_config["top1"]})
        trainer = types.SimpleNamespace()
        cb.setup(trainer, pl_module, stage="validate")

        # Feed a batch
        labels = torch.randint(0, C, (B,))
        batch = {"image": torch.empty(B, 3, 224, 224), "label": labels}
        get_mock.side_effect = lambda key, b, o, caller_name=None: b[key]

        # Run
        cb.on_validation_batch_end(
            trainer, pl_module, outputs={}, batch=batch, batch_idx=0
        )

        # Callback stores logits in batch under <name>_preds
        pred_key = f"{cb.name}_preds"
        got = batch[pred_key]  # [B, C]

        # Reference: cosine(img_raw, cls_raw) = (normalize both) dot
        ref = (
            torch.nn.functional.normalize(img_raw, dim=-1)
            @ torch.nn.functional.normalize(cls_raw, dim=-1).T
        )

        torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-6)
        assert got.shape == (B, C)

    @patch("stable_pretraining.callbacks.clip_zero_shot.get_data_from_batch_or_outputs")
    @patch("stable_pretraining.callbacks.clip_zero_shot.format_metrics_as_dict")
    def test_setup_builds_cached_class_embeds(self, fmt_mock, get_mock, class_setup):
        from stable_pretraining.callbacks.clip_zero_shot import CLIPZeroShot

        # Patch helpers: format_metrics_as_dict returns our metrics map
        fmt_mock.return_value = {"_train": {}, "_val": class_setup["metrics"]}

        cb = CLIPZeroShot(
            name="zs",
            image_key="image",
            class_key="label",
            class_names=class_setup["class_names"],
            image_backbone=class_setup["vision"],
            text_backbone=class_setup["text"],
            tokenizer_fn=class_setup["tokenizer_fn"],
            metrics=class_setup["metrics"],
        )
        pl_module = self._fake_pl_module({"top1": class_setup["metrics"]["top1"]})
        trainer = types.SimpleNamespace()  # unused

        cb.setup(trainer, pl_module, stage="validate")

        # class tokens computed once
        class_setup["tokenizer_fn"].assert_called_once()
        assert hasattr(cb, "class_embeds")
        assert cb.class_embeds.shape == (class_setup["C"], class_setup["D"])
        # normalized cache
        norms = cb.class_embeds.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    @patch("stable_pretraining.callbacks.clip_zero_shot.get_data_from_batch_or_outputs")
    @patch("stable_pretraining.callbacks.clip_zero_shot.format_metrics_as_dict")
    def test_val_batch_updates_metric_and_logs(self, fmt_mock, get_mock, class_setup):
        from stable_pretraining.callbacks.clip_zero_shot import CLIPZeroShot

        fmt_mock.return_value = {"_train": {}, "_val": class_setup["metrics"]}

        cb = CLIPZeroShot(
            name="zs",
            image_key="image",
            class_key="label",
            class_names=class_setup["class_names"],
            image_backbone=class_setup["vision"],
            text_backbone=class_setup["text"],
            tokenizer_fn=class_setup["tokenizer_fn"],
            metrics=class_setup["metrics"],
        )
        pl_module = self._fake_pl_module({"top1": class_setup["metrics"]["top1"]})
        trainer = types.SimpleNamespace()

        cb.setup(trainer, pl_module, stage="validate")

        # Build a batch: labels determine which class embedding we "see"
        B = 16
        labels = torch.randint(0, class_setup["C"], (B,))
        images = torch.randn(B, 3, 224, 224)  # ignored by DummyVision
        batch = {"image": images, "label": labels}

        # get_data_from_batch_or_outputs returns straight from batch
        def _get(key, batch, outputs, caller_name):
            return batch[key]

        get_mock.side_effect = _get

        # Feed labels to DummyVision so it can index class_embeds
        class_setup["vision"].labels = labels

        cb.on_validation_batch_end(
            trainer, pl_module, outputs={}, batch=batch, batch_idx=0
        )

        # Metric state should reflect perfect predictions => acc ≈ 1.0
        acc = class_setup["metrics"]["top1"].compute().item()
        assert acc == pytest.approx(1.0, rel=0, abs=1e-6)

        # Logged keys exist with expected prefix
        # (Lightning would serialize the metric; here we just see class names)
        assert any(k.startswith("val/zs_top1") for k in pl_module.logged.keys())

    @patch("stable_pretraining.callbacks.clip_zero_shot.get_data_from_batch_or_outputs")
    @patch("stable_pretraining.callbacks.clip_zero_shot.format_metrics_as_dict")
    def test_accepts_list_or_tensor_labels(self, fmt_mock, get_mock, class_setup):
        from stable_pretraining.callbacks.clip_zero_shot import CLIPZeroShot

        fmt_mock.return_value = {"_train": {}, "_val": class_setup["metrics"]}
        cb = CLIPZeroShot(
            name="zs",
            image_key="image",
            class_key="label",
            class_names=class_setup["class_names"],
            image_backbone=class_setup["vision"],
            text_backbone=class_setup["text"],
            tokenizer_fn=class_setup["tokenizer_fn"],
            metrics=class_setup["metrics"],
        )
        pl_module = self._fake_pl_module({"top1": class_setup["metrics"]["top1"]})
        trainer = types.SimpleNamespace()

        cb.setup(trainer, pl_module, stage="validate")

        # Labels as Python list
        labels = [0, 1, 2, 3]
        images = torch.randn(4, 3, 224, 224)
        batch = {"image": images, "label": labels}

        def _get(key, batch, outputs, caller_name):
            return batch[key]

        get_mock.side_effect = _get
        class_setup["vision"].labels = torch.tensor(labels)

        cb.on_validation_batch_end(
            trainer, pl_module, outputs={}, batch=batch, batch_idx=0
        )

        # Should still update and compute valid accuracy
        acc = class_setup["metrics"]["top1"].compute().item()
        assert 0.0 <= acc <= 1.0

    def test_initializer_logs_and_maps(self, class_setup, capsys):
        # Smoke-test ctor: just ensure it doesn't throw and records the class map
        from stable_pretraining.callbacks.clip_zero_shot import CLIPZeroShot

        cb = CLIPZeroShot(
            name="zs",
            image_key="image",
            class_key="label",
            class_names=class_setup["class_names"],
            image_backbone=class_setup["vision"],
            text_backbone=class_setup["text"],
            tokenizer_fn=class_setup["tokenizer_fn"],
            metrics={
                "top1": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=class_setup["C"]
                )
            },
        )
        assert cb.class_map[0] == class_setup["class_names"][0]

    @patch("stable_pretraining.callbacks.clip_zero_shot.get_data_from_batch_or_outputs")
    @patch("stable_pretraining.callbacks.clip_zero_shot.format_metrics_as_dict")
    def test_early_return_no_image(self, fmt_mock, get_mock, class_setup):
        from stable_pretraining.callbacks.clip_zero_shot import CLIPZeroShot

        fmt_mock.return_value = {"_train": {}, "_val": class_setup["metrics"]}

        cb = CLIPZeroShot(
            name="zs",
            image_key="image",
            class_key="label",
            class_names=class_setup["class_names"],
            image_backbone=class_setup["vision"],
            text_backbone=class_setup["text"],
            tokenizer_fn=class_setup["tokenizer_fn"],
            metrics=class_setup["metrics"],
        )
        pl_module = self._fake_pl_module({"top1": class_setup["metrics"]["top1"]})
        trainer = types.SimpleNamespace()

        cb.setup(trainer, pl_module, stage="validate")

        # Return None for image -> early return
        def _get(key, batch, outputs, caller_name=None):
            return None if key == "image" else batch.get(key)

        get_mock.side_effect = _get

        # Spy on metric.update: it should NOT be called
        metric = class_setup["metrics"]["top1"]
        with patch.object(metric, "update", wraps=metric.update) as spy_update:
            cb.on_validation_batch_end(
                trainer, pl_module, outputs={}, batch={}, batch_idx=0
            )
            spy_update.assert_not_called()
            # also nothing logged
            assert pl_module.logged == {}

    @patch("stable_pretraining.callbacks.clip_zero_shot.get_data_from_batch_or_outputs")
    @patch("stable_pretraining.callbacks.clip_zero_shot.format_metrics_as_dict")
    def test_similarity_math_nonperfect(self, fmt_mock, get_mock, class_setup):
        from stable_pretraining.callbacks.clip_zero_shot import CLIPZeroShot

        fmt_mock.return_value = {"_train": {}, "_val": class_setup["metrics"]}

        # Build slightly noisy image features so accuracy is high but not 1.0
        torch.manual_seed(0)
        C, D = class_setup["C"], class_setup["D"]
        labels = torch.arange(C) % C
        noise = 0.1 * torch.randn(C, D)
        img_feats = F.normalize(class_setup["class_embeds"] + noise, dim=-1)

        class PlainVision(nn.Module):
            def __init__(self, feats):
                super().__init__()
                self.feats = feats

            def forward(self, pixel_values=None):
                return types.SimpleNamespace(image_embeds=self.feats)

        vision = PlainVision(img_feats)
        text = class_setup["text"]

        cb = CLIPZeroShot(
            name="zs",
            image_key="image",
            class_key="label",
            class_names=class_setup["class_names"],
            image_backbone=vision,
            text_backbone=text,
            tokenizer_fn=lambda _: torch.ones(C, 4, dtype=torch.long),
            metrics={
                "top1": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=C, top_k=1, average="micro"
                )
            },
        )
        pl_module = self._fake_pl_module({"top1": cb.metrics_config["top1"]})
        trainer = types.SimpleNamespace()
        cb.setup(trainer, pl_module, stage="validate")

        batch = {"image": torch.empty(C, 3, 224, 224), "label": labels}
        get_mock.side_effect = lambda key, batch, outputs, caller_name=None: batch[key]

        cb.on_validation_batch_end(
            trainer, pl_module, outputs={}, batch=batch, batch_idx=0
        )

        metric = pl_module.callbacks_metrics["zs"]["_val"]["top1"]
        acc = metric.compute().item()
        assert 0.8 <= acc <= 1.0
