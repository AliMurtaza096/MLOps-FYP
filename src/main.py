
import hydra
from src.model import RandomForest,XGBoost
from src.data import Preprocessing
from src.experiment_tracking import ModelSelection, MLFlowTracker
from src.train import Training
from hydra.core.config_store import ConfigStore
from src.config_dir.config_struct import ChurnConfig
import pandas as pd
cs = ConfigStore.instance()
cs.store(name="churn_config", node=ChurnConfig)


@hydra.main(config_path='src/config_dir', config_name='config')
def main(data_path,cfg=ChurnConfig):
    tracker = MLFlowTracker('demo_exp1',
                            '/home/ali/Desktop/FYP/MLflow/mlops_fyp/mlruns')
    print(data_path)
    new_data_read = pd.read_csv(data_path)
    previous_data_read=  pd.read_csv('/home/ali/Desktop/FYP/MLflow/mlops_fyp/dataset/train/Churn_Modelling.csv')
    
    #Data Merger
    merged_data = pd.concat([previous_data_read,new_data_read],axis='rows')
    merged_data.to_csv('/home/ali/Desktop/FYP/MLflow/mlops_fyp/dataset/train/files/merged.csv')  
    
    preprocessing = Preprocessing()
    pre_processed_dataset = preprocessing.preprocess(
        0.2,merged_data)

    # Random Forest Model Creation
    model= RandomForest()
    # model = RandomForest(cfg.params.n_estimators, cfg.params.criterion)
    model = model.create_model()
    # tracker.log_params(cfg.params.n_estimators,cfg.params.criterion)
    best_selected_model = Training(
        model, pre_processed_dataset, tracker, 'accuracy')
    best_selected_model.train()







if __name__ == "__main__":
    main()
