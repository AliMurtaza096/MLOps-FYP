
import hydra
from model import RandomForest,XGBoost
from data import Preprocessing
from experiment_tracking import ModelSelection, MLFlowTracker
from train import Training
from hydra.core.config_store import ConfigStore
from config_dir.config_struct import ChurnConfig
import pandas as pd


class Retrain():
    def __init__(self,data_path):
        self.data_path= data_path
    @hydra.main(config_path='config_dir', config_name='config')
    def retrain_model(self,cfg=ChurnConfig      ):
        tracker = MLFlowTracker(cfg.names.experiment_name,
                                cfg.paths.mlflow_tracking_uri)
        
        
        #Data Merger
        
        previous_data_read = pd.read_csv(cfg.paths.train_dir)
        new_data_read = pd.read_csv(self.data_path)
        
        merged_data = pd.concat([previous_data_read,new_data_read],axis='rows')
        
        
        preprocessing = Preprocessing()
        pre_processed_dataset = preprocessing.preprocess(
            cfg.params.train_test_split,merged_data)
        
        model= RandomForest()
        # model = RandomForest(cfg.params.n_estimators, cfg.params.criterion)
        model = model.create_model()
        # tracker.log_params(cfg.params.n_estimators,cfg.params.criterion)
        best_selected_model = Training(
            model, pre_processed_dataset, tracker, cfg.names.metric_name)
        best_selected_model.train()