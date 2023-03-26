from dataclasses import dataclass


@dataclass
class Paths:
    train_dir: str
    mlflow_tracking_uri: str
    model_artifactory_dir: str


@dataclass
class Params:
    n_estimators: int
    criterion: str
    train_test_split: str


@dataclass
class Names:
    experiment_name: str
    metric_name: str


@dataclass
class ChurnConfig:
    paths: Paths
    params: Params
    names: Names
