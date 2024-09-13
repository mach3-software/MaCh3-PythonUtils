import toml
import argparse

from file_handling.chain_handler import ChainHandler
from machine_learning.ml_factory import MLFactory
from machine_learning.scikit_interface import SciKitInterface
import scipy.stats as stats

if __name__=="__main__":
        
    parser = argparse.ArgumentParser(usage="python make_plots -c <config_name>.toml")
    parser.add_argument("-c", "--config", help="TOML config file", required=True)

    args = parser.parse_args()
    
    toml_config = toml.load(args.config)    
    
    # Process MCMC chain    
    file_handler = ChainHandler(toml_config["FileSettings"]["FileName"],
                                toml_config["FileSettings"]["ChainName"],
                                toml_config["FileSettings"]["Verbose"])
    
    file_handler.ignore_plots(toml_config["FileSettings"]["IgnoredParameters"])
    file_handler.add_additional_plots(toml_config["FileSettings"]["ParameterNames"])
    file_handler.add_additional_plots(toml_config["FileSettings"]["LabelName"], True)

    file_handler.add_new_cuts(toml_config["FileSettings"]["ParameterCuts"])

    file_handler.convert_ttree_to_array()
        
    factory = MLFactory(file_handler, toml_config["FileSettings"]["LabelName"])
    if toml_config["FitterSettings"]["FitterPackage"].lower() == "scikit":        
        interface = factory.setup_scikit_model(toml_config["FitterSettings"]["FitterName"],
                                   **toml_config["FitterSettings"]["FitterKwargs"])
        
    else:
        raise ValueError("Input not recognised!")
    
    if toml_config["FitterSettings"].get("AddFromExternalModel"):
        external_model = toml_config["FitterSettings"]["ExternalModel"]
        interface.load_model(external_model)
    
    interface.set_training_test_set(toml_config["FitterSettings"]["TestSize"])
    
    interface.train_model()
    interface.test_model()
    interface.save_model(toml_config["FileSettings"]["ModelOutputName"])