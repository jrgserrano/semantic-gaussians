from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from opensplat3d.params import (
    ClusterParams,
    DescriptionParams,
    ExportScanNetppParams,
    LanguageParams,
    ModelParams,
    OptimizationParams,
    PipeParams,
)


@dataclass
class Config:
    model: ModelParams = field(default_factory=lambda: ModelParams())
    pipe: PipeParams = field(default_factory=lambda: PipeParams())
    opt: OptimizationParams = field(default_factory=lambda: OptimizationParams())
    cluster: ClusterParams = field(default_factory=lambda: ClusterParams())
    lang: LanguageParams = field(default_factory=lambda: LanguageParams())
    export_scannetpp: ExportScanNetppParams = field(
        default_factory=lambda: ExportScanNetppParams()
    )
    desc: DescriptionParams = field(default_factory=lambda: DescriptionParams())


def to_dict(config: Config) -> dict:
    return OmegaConf.to_container(config, resolve=True)  # type: ignore


def load_config(overrides: list[str], config_file: str | None = None) -> Config:
    default_config: DictConfig = OmegaConf.structured(Config)
    if config_file is not None:
        default_config.merge_with(OmegaConf.load(config_file))
    default_config.merge_with_dotlist(overrides)
    return default_config  # type: ignore


def save_config(config: Config, overrides: list[str] | None, output_dir: Path):
    OmegaConf.save(config, output_dir / "config.yaml", resolve=True)
    if overrides is not None and len(overrides) > 0:
        OmegaConf.save(
            OmegaConf.create(overrides), output_dir / "overrides.yaml", resolve=True
        )


def config_from_yaml(config_path: Path) -> Config:
    default_config: DictConfig = OmegaConf.structured(Config)
    default_config.merge_with(OmegaConf.load(config_path))
    return default_config  # type: ignore
