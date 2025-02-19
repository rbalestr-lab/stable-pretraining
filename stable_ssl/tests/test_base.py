import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf


def test_base_trainer(tmp_path):
    with initialize(version_base="1.2", config_path="configs"):
        cfg = compose(config_name="tiny_mnist")

        # Write out the config.yaml to simulate Hydra’s usual job folder
        hydra_dir = tmp_path / ".hydra"
        hydra_dir.mkdir(parents=True, exist_ok=True)
        config_file = hydra_dir / "config.yaml"
        OmegaConf.save(config=cfg, f=config_file)

        # Convert to a dictionary to easily add missing keys
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict["trainer"]["logger"]["dump_path"] = tmp_path
        cfg = OmegaConf.create(cfg_dict)

        trainer = hydra.utils.instantiate(
            cfg.trainer, _convert_="object", _recursive_=False
        )
        print(trainer)
        # trainer.setup()
        # trainer.launch()

        # logs = list(tmp_path.glob("logs_rank_*.jsonl"))
        # assert len(logs) > 0, "No logs were created!"

        # ckpts = list(tmp_path.glob("*.ckpt"))
        # assert len(ckpts) >= 1, "No .ckpt files found in the output!"
