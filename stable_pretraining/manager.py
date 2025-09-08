import copy
import json
import signal
from datetime import timedelta
from pathlib import Path
from typing import Union

import hydra
import lightning
import lightning as pl
import pandas as pd
import submitit
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf, open_dict

import wandb

from .utils import get_required_fn_parameters


class Manager(submitit.helpers.Checkpointable):
    """Manages training with logging, scheduling, and checkpointing support.

    Args:
        trainer (Union[dict, DictConfig, pl.Trainer]): PyTorch Lightning trainer configuration or instance.
        module (Union[dict, DictConfig, pl.LightningModule]): Lightning module configuration or instance.
        data (Union[dict, DictConfig, pl.LightningDataModule]): Data module configuration or instance.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        ckpt_path (str, optional): Path to checkpoint for resuming training. Defaults to None.
    """

    def __init__(
        self,
        trainer: Union[dict, DictConfig, pl.Trainer],
        module: Union[dict, DictConfig, pl.LightningModule],
        data: Union[dict, DictConfig, pl.LightningDataModule],
        seed: int = None,
        ckpt_path: str = "last",
    ):
        # This is the state that will be saved by `checkpoint`
        # we do deepcopy in case the user changes things after
        # padding the dicts (since it will be a copy by reference)
        if seed is None:
            logging.warning(
                "User didn't specify seed, runs won't be exactly reproducible!"
            )

        if type(trainer) is dict:
            trainer = OmegaConf.create(trainer)
        if type(trainer) is DictConfig:
            self.trainer: DictConfig = copy.deepcopy(trainer)
            logging.debug("\t● trainer config saved ✅")
        elif isinstance(trainer, pl.Trainer):
            self.trainer = trainer
            logging.debug("\t● trainer already instantiated ✅")
        else:
            raise ValueError(
                f"`trainer` must be a dict, DictConfig or pl.Trainer, not {type(trainer)}"
            )

        if type(module) is dict:
            module = OmegaConf.create(module)
        if type(module) is DictConfig:
            self.module: DictConfig = copy.deepcopy(module)
            logging.debug("\t● module config saved ✅")
        elif isinstance(module, pl.LightningModule):
            self.module = module
            logging.debug("\t● module already instantiated ✅")
        else:
            raise ValueError(
                f"`module` must be a dict, DictConfig or pl.LightningModule, not {type(module)}"
            )

        if type(data) is dict:
            data = OmegaConf.create(data)
        if type(data) is DictConfig:
            self.data: DictConfig = copy.deepcopy(data)
            logging.debug("\t● data config saved ✅")
        elif isinstance(data, pl.LightningDataModule):
            self.data = data
            logging.debug("\t● data already instantiated ✅")
        else:
            raise ValueError(
                f"`data` must be a dict, DictConfig or pl.LightningDataModule, not {type(data)}"
            )

        self.seed = seed
        self.ckpt_path = ckpt_path
        # self.slurm_requeue_signal = slurm_requeue_signal

    @rank_zero_only
    def init_and_sync_wandb(self):
        # only useful with wandb
        # to set any non-given variable to the DictConfig
        # so that we can resume on requeue

        # this override is only useful if receiving parameters
        # from wandb e.g. using wandb sweep. We retrieve them and
        # requeue with those instead of user args
        self.override = []
        if isinstance(
            self.trainer.logger, lightning.pytorch.loggers.logger.DummyLogger
        ):
            logging.info("📈📈📈 DummyLogger already setup, skipping init 📈📈📈")
            return
        elif isinstance(
            self.trainer.logger, lightning.pytorch.loggers.tensorboard.TensorBoardLogger
        ):
            logging.info("📈📈📈 TensorBoardLogger already setup, skipping init 📈📈📈")
            logging.info(f"📈📈📈 root_dir={self.trainer.logger.root_dir} 📈📈📈")
            logging.info(f"📈📈📈 save_dir={self.trainer.logger.save_dir} 📈📈📈")
            logging.info(f"📈📈📈 log_dir={self.trainer.logger.log_dir} 📈📈📈")
            return
        elif isinstance(
            self.trainer.logger, lightning.pytorch.loggers.csv_logs.CSVLogger
        ):
            logging.info("📈📈📈 CSVLogger already setup, skipping init 📈📈📈")
            logging.info(f"📈📈📈 root_dir={self.trainer.logger.root_dir} 📈📈📈")
            logging.info(f"📈📈📈 save_dir={self.trainer.logger.save_dir} 📈📈📈")
            logging.info(f"📈📈📈 log_dir={self.trainer.logger.log_dir} 📈📈📈")
            return
        elif isinstance(
            self.trainer.logger, lightning.pytorch.loggers.wandb.WandbLogger
        ):
            logging.info("📈📈📈 WandbLogger already setup, skipping init 📈📈📈")
            logging.info(f"📈📈📈 init={self.trainer.logger._wandb_init} 📈📈📈")
            return
        elif self.trainer.logger is None:
            logging.warning("📈📈📈 No logger used! 📈📈📈")
            return

        if "wandb" not in self.trainer.logger._target_.lower():
            return
        logging.info("📈📈📈 Using Wandb 📈📈📈")
        exp = self._trainer.logger.experiment
        with open_dict(self.trainer):
            prefix = "\t\t● config's "
            for name in ["entity", "project", "name", "id"]:
                cfg_value = self.trainer.logger.get(name, None)
                w_value = getattr(exp, name)
                if cfg_value == w_value:
                    logging.info(f"{prefix}{name} ({cfg_value}) left as-is ✅")
                    continue
                self.override.append(f"++manager.trainer.logger.{name}={w_value}")
                logging.info(f"{prefix}{name} ({cfg_value}) updated to {w_value} ✅")
                # setattr(self.trainer.logger, name, w_value)
                self.trainer.logger[name] = w_value
            if self.trainer.logger.get("resume", None):
                logging.info(f"{prefix}`resume` already set to `allow`! ✅")
            else:
                self.override.append("++manager.trainer.logger.resume=allow")
                self.trainer.logger.resume = "allow"
                logging.info(f"{prefix}`resume` set to `allow` for subsequent runs ✅")

        # we defer adding the config to later
        # to make sure we use the possibly given
        # sweep config
        # if self.logger.get("config", None) is not None:
        #     # this will be a nested dict
        #     config = OmegaConf.to_container(self.logger.config, resolve=True)
        #     # now we flatten
        #     config = pd.json_normalize(config, sep="_")
        #     config = config.to_dict(orient="records")[0]
        #     logging.info(f"\tflattening Hydra's config for Wandb ✅")
        #     self.logger.config = None
        # else:
        #     config = None

        if exp.offline:
            previous_run = self._wandb_previous_dir()
            logging.info(f"\t\tFound a previous run ({previous_run}), reusing config")
            with open(previous_run / "files/wandb-config.json", "r") as f:
                last_config = json.load(f)
            # at most last_config has an extra `ckpt_path`
            exp.config.update(last_config)
            logging.info("\t\treloaded!")
        elif len(wandb.config.keys()):
            logging.info("\t\ta Wandb™ config is provided, not uploading Hydra's:")
            # TODO: make Wandb parameters the trainer one
            # for key, value in wandb.config.items():
            #     # need to handle the fact that our base configs have a _
            #     # and users wouldn't provide that
            #     accessor = key.split(".")
            #     if accessor[0] == "trainer":
            #         accessor = accessor[1:]
            #     if accessor[0] in [
            #         "data",
            #         "module",
            #         "hardware",
            #         "loss",
            #         "metric",
            #         "optim",
            #     ]:
            #         if "_" != accessor[0][0]:
            #             accessor[0] = "_" + accessor[0]
            #     key = ".".join(accessor)
            #     try:
            #         original = rgetattr(self, key)
            #         rsetattr(self, key, value)
            #         assert rgetattr(self, key) == value
            #         logging.info(
            #             f"\t\t\toverriding: {key} from {original} to {value} ✅"
            #         )
            #     except Exception as e:
            #         logging.error(f"❌ Error while trying to override {key} ❌")
            #         raise e
        else:
            logging.info("\tWandb's config is empty, using Hydra's 📤")
            config = dict(
                trainer=OmegaConf.to_container(self.trainer, resolve=True),
                module=OmegaConf.to_container(self.module, resolve=True),
                data=OmegaConf.to_container(self.data, resolve=True),
            )
            config = pd.json_normalize(config, sep=".")
            config = config.to_dict(orient="records")[0]
            while True:
                logging.info("\t\tflattening one level of Hydra's config) 📤")
                valid = True
                for k in list(config.keys()):
                    if type(config[k]) is list:
                        valid = False
                        for i, j in enumerate(config[k]):
                            config[f"{k}.{i}"] = j
                        del config[k]
                config = pd.json_normalize(config, sep=".")
                config = config.to_dict(orient="records")[0]
                if valid:
                    break
            logging.info(f"\tFinal Hydra's config has {len(config)} items) 📤")
            wandb.config.update(config)
            # TODO: should we updated the config to the DictConfig too for next run to check?
            # with open_dict(self.logger):
            #     self.trainer.logger.config = config

    @property
    def instantiated_module(self):
        if not isinstance(self.module, pl.LightningModule):
            logging.info("\t● instantiating pl_module...")
            # with self._trainer.init_module():
            self._instantiated_module = hydra.utils.instantiate(
                self.module, _convert_="object"
            )
            logging.info("\t● module instantiated ✅")
        else:
            self._instantiated_module = self.module
        return self._instantiated_module

    @property
    def instantiated_data(self):
        if not isinstance(self.data, pl.LightningDataModule):
            self._instantiated_data = hydra.utils.instantiate(
                self.data, _convert_="object", _recursive_=False
            )
            logging.info("\t● data instantiated ✅")
        else:
            self._instantiated_data = self.data
        return self._instantiated_data

    def __call__(self):
        # self._setup_logging()
        logging.info(f"📁📁📁 CURRENT WORKING DIR: {Path().resolve()} 📁📁📁")

        # if "SLURM_JOB_ID" in os.environ:
        #     # single-node and multi-node distributed training on SLURM cluster
        #     # requeue job on SLURM preemption
        #     self.submitit_signal = signal.getsignal(
        #         signal.__dict__[self.slurm_requeue_signal]
        #     )
        #     logging.info(f"\t● saved signal {self.submitit_signal} ✅")
        #     logging.info(
        #         f"\t● setting up checkpoint and requeue on {self.slurm_requeue_signal} ✅"
        #     )
        #     signal.signal(
        #         signal.__dict__[self.slurm_requeue_signal], self.checkpoint_and_requeue
        #     )
        logging.info(f"🌱🌱🌱 SEEDING EVERYTHING with {self.seed=} 🌱🌱🌱")
        pl.seed_everything(self.seed, workers=True)
        if isinstance(self.trainer, pl.Trainer):
            self._trainer = self.trainer
        else:
            if "callbacks" in self.trainer:
                logging.info("\t● instantiating callbacks...")
                callbacks = hydra.utils.instantiate(
                    self.trainer.callbacks, _convert_="object"
                )
                for i, callback in enumerate(callbacks):
                    if not callable(callback):
                        continue
                    assert ["pl_module"] == get_required_fn_parameters(callback)
                    callbacks[i] = callback(pl_module=self.instantiated_module)
                logging.info("\t● callbacks instantiated ✅")
                del self.trainer.callbacks

            else:
                callbacks = []

            # we use the following partial to give our init callbacks manually since otherwise
            # hydra instantiate throws an error
            self._trainer = hydra.utils.instantiate(
                self.trainer, _convert_="object", _partial_=True
            )
            self._trainer = self._trainer(callbacks=callbacks)
            if not isinstance(self._trainer, pl.Trainer):
                raise ValueError("`trainer` should be a Trainer")
            logging.info("\t● trainer instantiated ✅")
        self.init_and_sync_wandb()
        logging.info("\t● logger updated accordingly ✅")

        logging.info("\t● 👂👂👂 SIGNALS HANDLERS 👂👂👂")
        logging.info(f"\t\t- SIGUSR1: `{signal.getsignal(signal.SIGUSR1)}`")
        logging.info(f"\t\t- SIGUSR2: `{signal.getsignal(signal.SIGUSR2)}`")
        logging.info(f"\t\t- SIGCONT: `{signal.getsignal(signal.SIGCONT)}`")
        logging.info(f"\t\t- SIGTERM: `{signal.getsignal(signal.SIGTERM)}`")

        # when using submitit launcher, Hydra uses its own checkpoint method
        # https://github.com/facebookresearch/hydra/blob/main/plugins/hydra_submitit_launcher/hydra_plugins/hydra_submitit_launcher/submitit_launcher.py#L78
        # we replace it with ours
        # https://github.com/facebookresearch/hydra/issues/2042
        # https://github.com/facebookincubator/submitit/blob/main/submitit/core/job_environment.py#L212
        fn = signal.getsignal(signal.SIGUSR2)

        if hasattr(fn, "__self__"):
            self._hydra_self = fn.__self__._delayed.function.checkpoint.__self__
            fn.__self__._delayed.function.checkpoint = self.checkpoint
        else:
            self._hydra_self = self

        # logging.info(f"\t● Searching for checkpoint to warm restart...")
        # ckpt_path = None
        # if wandb.run and not wandb.run.offline:
        #     logging.info(
        #         f"\t\t● Wandb is online... searching for `requeue_checkpoint` in Artifacts..."
        #     )
        #     r = wandb.Api().run(wandb.run.path)
        #     artifacts = r.logged_artifacts()
        #     logging.info(f"\t\t● wandb run artifacts:")
        #     for artifact in artifacts:
        #         logging.info(
        #             f"\t\t\t● {artifact.name}, {artifact.type}, {artifact.created_at}"
        #         )
        #         if artifact.name.split(":")[0] == "requeue_checkpoint":
        #             logging.info(f"\t\t● Checkpoint found! 🔥")
        #             ckpt_path = artifact
        #     # we wait the end to download to make sure we get the
        #     # latest version
        #     if ckpt_path:
        #         datadir = Path(ckpt_path.download()) / "checkpoint.ckpt"
        #         logging.info(f"\t● Checkpoint downloaded ({datadir})!  🔥")
        #         ckpt_path = datadir
        #     else:
        #         logging.info(
        #             f"\t\t● No Checkpoint artifact found in Wandb artifacts... searching in config ❌"
        #         )
        #         if "ckpt_path" in wandb.run.config:
        #             logging.info(
        #                 f"\t\t● `ckpt_path` found in Wandb config: {ckpt_path} 🔥"
        #             )
        #             ckpt_path = wandb.run.config["ckpt_path"]
        #             if ckpt_path is not None and Path(ckpt_path).is_file():
        #                 logging.info(
        #                     f"\t\t● `ckpt_path` found in Wandb config: {ckpt_path} 🔥"
        #                 )
        #             else:
        #                 logging.info(
        #                     f"\t\t● `{ckpt_path=}` is not a valid file... not using it ❌"
        #                 )
        #                 ckpt_path = None
        #         else:
        #             logging.info(
        #                 f"\t\t● `ckpt_path` not found in online Wandb config ...!"
        #             )
        # else:
        #     logging.info(
        #         f"\t\t● Wandb is offline... searching in local logger's config..."
        #     )
        #     cfg = self.trainer.logger.get("config", {})
        #     if cfg and cfg.get("ckpt_path", None):
        #         ckpt_path = cfg["ckpt_path"]
        #         if ckpt_path is not None and Path(ckpt_path).is_file():
        #             logging.info(
        #                 f"\t\t● `ckpt_path` found in local config: {ckpt_path} 🔥"
        #             )
        #         else:
        #             logging.info(
        #                 f"\t\t● `{ckpt_path=}` is not a valid file... not using it ❌"
        #             )
        #             ckpt_path = None
        # if ckpt_path is None:
        #     logging.error(f"\t\t● No checkpoint found! ❌")
        logging.info("\t● 📞📞📞 CALLBACKS 📞📞📞")
        for c in self._trainer.checkpoint_callbacks:
            logging.info(c)
        for c in self._trainer.early_stopping_callbacks:
            logging.info(c)
        if Path("requeue_checkpoint.ckpt").is_file():
            if self.ckpt_path is not None:
                logging.warning(
                    "We are overring user given {self.ckpt_path}"
                    " to `requeue_checkpoint.ckpt`"
                )
            ckpt_path = "requeue_checkpoint.ckpt"
        else:
            ckpt_path = self.ckpt_path
        logging.info(f"📣📣📣 CALLING trainer.fit with {ckpt_path=} 📣📣📣")
        self._trainer.fit(
            self.instantiated_module,
            datamodule=self.instantiated_data,
            ckpt_path=ckpt_path,
        )
        self._dump_wandb_data()
        if Path("requeue_checkpoint.ckpt").is_file():
            logging.info("Cleaning up requeue checkpoint")
            Path("requeue_checkpoint.ckpt").unlink()

    def validate(self):
        logging.info("📣📣📣 CALLING trainer.validate 📣📣📣")

        self._trainer.validate(
            self.instantiated_module, datamodule=self.instantiated_data
        )
        self._dump_wandb_data()

    def predict(self):
        logging.info("📣📣📣 CALLING trainer.predict 📣📣📣")

        self._trainer.predict(
            self.instantiated_module, datamodule=self.instantiated_data
        )
        self._dump_wandb_data()

    def test(self):
        logging.info("📣📣📣 CALLING trainer.test 📣📣📣")

        self._trainer.test(self.instantiated_module, datamodule=self.instantiated_data)
        self._dump_wandb_data()
        # wandb.finish()
        # logging.info(f"closing wandb 🗑️")
        # cfg = wandb.run.config.as_dict()
        # return cfg, module.info

    @rank_zero_only
    def _dump_wandb_data(self):
        if wandb.run is None or not wandb.run.offline:
            return

        # Print the summary
        logging.info("Summary:")
        summary_dict = wandb.run.summary._as_dict()
        logging.info(json.dumps(summary_dict, indent=2))
        fname = Path(wandb.run.dir) / "wandb-summary.json"
        if fname.is_file():
            raise RuntimeError(f"Summary file already exists {fname}")
        with open(fname, "w") as f:
            json.dump(summary_dict, f)
        logging.info(f"\t● Saved summary at {fname} ✅")
        fname = Path(wandb.run.dir) / "wandb-config.json"
        if fname.is_file():
            raise RuntimeError(f"Config file already exists {fname}")
        with open(fname, "w") as f:
            json.dump(wandb.run.config.as_dict(), f)
        logging.info(f"\t● Saved config at {fname} ✅")

    def _wandb_previous_dir(self):
        # to remove the /files
        path = Path(wandb.run.dir).parent
        logging.info(f"\t\t● fetching previous Wandb runs from {path.parent} ✅")
        # this will be of the form
        # offline-run-20250413_025716-p8117tgi
        runs = list(path.parent.glob(f"offline-run-*-{wandb.run.id}"))
        logging.info(f"\t\t● found {len(runs)} run(s):")
        runs = sorted(runs)
        for run in runs:
            logging.info(f"\t\t\t● {run.name}")
        assert runs[-1] == path
        if len(runs) == 1:
            return None
        return runs[-2]

    def save_checkpoint(self, path: str = None, upload_wandb: bool = False):
        # TODO: figure out how to flush logging in subprocess
        print("Entering checkpoint method", flush=True)
        if path is None:
            path = (Path() / "checkpoint.ckpt").resolve()
            print(f"\t● saving checkpoint to local path {path} ⏳", flush=True)
        else:
            path = Path(path)
            if not path.parent.is_dir():
                path.parent.mkdir(parents=True)
            print(f"\t● saving checkpoint to user's path {path} ⏳", flush=True)
        self._trainer.save_checkpoint(str(path))
        print("\t● checkpoint saved ✅", flush=True)
        if upload_wandb:
            self._upload_checkpoint_for_requeue(path)

    @rank_zero_only
    def _upload_checkpoint_for_requeue(self, ckpt_path):
        # if "ckpt_path" in wandb.run.config:
        #     ckpt_path = Path(wandb.run.config["ckpt_path"])
        #     print(f"\t● `ckpt_path` already in config, updating it!", flush=True)

        # else:
        #     ckpt_path = Path(wandb.run.dir) / "checkpoint.ckpt"
        #     print(f"\t● `ckpt_path` set to {ckpt_path}!", flush=True)

        if not wandb.run.offline:
            print("\t● Wandb used and online:", flush=True)
            artifact = wandb.Artifact("requeue_checkpoint", "model")
            artifact.add_file(str(ckpt_path))
            artifact.ttl = timedelta(days=30)
            print("\t\t● artifact created ✅", flush=True)
            wandb.run.log_artifact(artifact)
            print("\t\t● artifact logged ✅", flush=True)
            ckpt_path.unlink()
            print("\t\t● local checkpoint deleted ✅", flush=True)
        else:
            print("\t● Wandb used and offline:", flush=True)
            wandb.run.config.update({"ckpt_path": str(ckpt_path.resolve())})
            print("\t● `ckpt_path` added to Wandb config ✅", flush=True)
        # for offline case
        self._dump_wandb_data()

    # @rank_zero_only
    # def requeue(self, *args, **kwargs):

    #     print(f"\t● requeing! 🔥", flush=True)
    #     print(args, kwargs)
    #     self.submitit_signal(*args, **kwargs)

    @rank_zero_only
    def checkpoint(self, *args, **kwargs):
        print(
            "⚠️⚠️⚠️ only SLURM should use this function,"
            "users should use `save_checkpoint` ⚠️⚠️⚠️",
            flush=True,
        )
        assert len(kwargs) == 0
        assert type(args[0]) is list
        print("Original Hydra's overrides: ", args[0], flush=True)
        # TODO add a check to make sure we don't double override anything
        print("Adding our overrides: ", self.override, flush=True)
        args[0].extend(self.override)
        inst = self._hydra_self.__class__(**self._hydra_self.params)
        inst.config = self._hydra_self.config
        inst.hydra_context = self._hydra_self.hydra_context
        inst.task_function = self._hydra_self.task_function
        # print(self._hydra_checkpoint, flush=True)
        # return self._hydra_checkpoint(*args, **kwargs)
        # child = self.__class__(
        #     self.trainer, self.module, self.data, self.seed, self.ckpt_path
        # )
        print("Saving checkpoint to `requeue_checkpoint.ckpt`")
        self.save_checkpoint("requeue_checkpoint.ckpt", upload_wandb=False)
        print("Requeing with: ", int, args, kwargs, flush=True)
        return submitit.helpers.DelayedSubmission(inst, *args, **kwargs)

    # def checkpoint_and_requeue(self, *args, **kwargs):
    #     self.save_checkpoint()
    #     self.requeue(*args, **kwargs)
