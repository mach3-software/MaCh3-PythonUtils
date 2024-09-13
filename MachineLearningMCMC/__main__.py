import yaml
import argparse

from file_handling.chain_handler import ChainHandler
from machine_learning.ml_factory import MLFactory

if __name__=="__main__":
        
    parser = argparse.ArgumentParser(usage="python make_plots -c <config_name>.yaml")
    parser.add_argument("-c", "--config", help="yaml config file", required=True)

    args = parser.parse_args()
    
    with open(args.config, 'r') as c:
        yaml_config = pyyaml.load(c)    
    
    # Process MCMC chain    
    file_handler = ChainHandler(yaml_config["FileSettings"]["FileName"],
                                yaml_config["FileSettings"]["ChainName"],
                                yaml_config["FileSettings"]["Verbose"])
    
    file_handler.ignore_plots(yaml_config["FileSettings"]["IgnoredParameters"])
    file_handler.add_additional_plots(yaml_config["FileSettings"]["ParameterNames"])
    file_handler.add_additional_plots(yaml_config["FileSettings"]["LabelName"], True)

    file_handler.add_new_cuts(yaml_config["FileSettings"]["ParameterCuts"])

    file_handler.convert_ttree_to_array()
        
    factory = MLFactory(file_handler, yaml_config["FileSettings"]["LabelName"])
    if yaml_config["FitterSettings"]["FitterPackage"].lower() == "scikit":        
        interface = factory.setup_scikit_model(yaml_config["FitterSettings"]["FitterName"],
                                   **yaml_config["FitterSettings"]["FitterKwargs"])
        
    else:
        raise ValueError("Input not recognised!")
    
    if yaml_config["FitterSettings"].get("AddFromExternalModel"):
        external_model = yaml_config["FitterSettings"]["ExternalModel"]
        interface.load_model(external_model)
    
    interface.set_training_test_set(yaml_config["FitterSettings"]["TestSize"])
    
    interface.train_model()
    interface.test_model()
    interface.save_model(yaml_config["FileSettings"]["ModelOutputName"])