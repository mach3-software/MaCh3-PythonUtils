import toml
import argparse

from file_handling.chain_handler import ChainHandler
from machine_learning.ml_factory import MLFactory
from machine_learning.scikit_interface import SciKitInterface

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
    factory = MLFactory(file_handler, toml_config["FileSettings"]["LabelName"])
    
    if toml_config["FitterSettings"]["FitterPackage"].lower() == "scikit":        
        interface = factory.setup_scikit_model(toml_config["FitterSettings"]["FitterName"],
                                   **toml_config["FitterSettings"]["FitterKwargs"])
        
    else:
        raise ValueError("Input not recognised!")
    
    interface.set_training_test_set(toml_config["FitterSettings"]["TestSize"])
    
    interface.train_model()
    interface.train_model()