# MachineLearningMCMC

Very simple tool for predicting likelihoods from Markov Chains Using ML tools. Currently only accepts chains where all variables are saved within a ROOT TTree. 

# Configs
Configs are in TOML format

```toml
[FileSettings]
    FileName = "/path/to/file"  # Name of File
    ChainName = "/tree/name"    # Name of Tree in file containing chain

    # Names of parameters to fit, finds all parameters containing names in this string as sub-string
    ParameterNames = ["sin2th", "sin2th","delm2_12", "delta", "xsec"]

    # Name of variable you're fitting in
    LabelName = "LogL"
```
# Executables
Simply run `python MachineLearningMCMC -c /path/to/toml/config` and it'll automatically run the chain. 

# TODO:
* Flexible factory-based interface so any file can be used
* More libs than scikit
* Configurable fitters
* Clearer Readme