import os
import hydra
from trainer import Trainer
from omegaconf import OmegaConf
from rich.console import Console

console = Console()


@hydra.main(version_base="1.3", config_path="conf", config_name="trainer")
def main(cfg):
    upsample_tune_only: bool = cfg.get("upsample_tune_only", False)
    # console.print(OmegaConf.to_yaml(cfg, resolve=True))
    ckpt = cfg.get("ckpt", None)
    if not upsample_tune_only:
        if ckpt is not None:
            trainer = Trainer.load(cfg.ckpt, cfg)
        else:
            trainer = Trainer(cfg)
        trainer.train_loop()

        if hasattr(cfg, "upsample_tune") and cfg.upsample_tune.enabled == True:
            trainer.tune_with_upsample_model()
    else:
        assert (
            ckpt is not None
        ), "ckpt must be specified when upsample_tune_only is True"
        console.print("[red]Tune from ckpt: {}[/red]".format(ckpt))
        trainer = Trainer.load(cfg.ckpt, cfg)

        trainer.tune_with_upsample_model()

        return 0


if __name__ == "__main__":
    main()
