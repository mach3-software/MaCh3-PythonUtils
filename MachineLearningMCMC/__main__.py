import toml
import argparse

from file_handling.chain_handler import ChainHandler
from machine_learning.scikit_interface import SciKitInterface
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

if __name__=="__main__":
        
    parser = argparse.ArgumentParser(usage="python make_plots -c <config_name>.toml")
    parser.add_argument("-c", "--config", help="TOML config file")

    args = parser.parse_args()
    
    toml_config = toml.load(args.config)    
    
    # Process MCMC chain    
    file_handler = ChainHandler(toml_config["FileSettings"]["FileName"], toml_config["FileSettings"]["ChainName"])
    
    file_handler.add_additional_plots(toml_config["FileSettings"]["ParameterNames"])
    file_handler.add_additional_plots(toml_config["FileSettings"]["LabelName"], True)

    file_handler.convert_ttree_to_array()
    
    # Do some ML
    regressor = SciKitInterface(file_handler, toml_config["FileSettings"]["LabelName"])
    
    # Setup random forest  
#    random_forest = RandomForestRegressor(verbose=True, n_jobs=8)
    random_forest = GradientBoostingRegressor(verbose=True, n_estimators=300)
    
    regressor.add_model(random_forest)
    features, predictions = regressor.separate_dataframe()
    regressor.set_training_test_set(features, predictions, 0.40)
    
    regressor.train_model()
    regressor.test_model()