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
    
    # Parameters you don't want to include in the model
    IgnoredParameters = ["LogL_systematic_xsec_cov", "Log", "LogL_systematic_nddet_cov", ]

    # Any cuts, for MaCh3 I'd recommend capping LogL
    ParameterCuts = ["LogL<12345678", "step>10000"]

    # Do you want it to be verbose?
    Verbose = false

    # Where is the model being pickled?
    ModelOutputName="histboost_model_full.pkl"


[FitterSettings]
    # Package model is included in
    FitterPackage = "SciKit" 

    # Fitter Name
    FitterName = "HistBoost"

    # Size of test set (range is 0-1)
    TestSize = 0.8
    # Set fitter Hyper Parameters, these are found in the fitter's readme
    [FitterSettings.FitterKwargs]
            # n_jobs = 32
            verbose=true
            max_iter = 10000
            # n_estimators = 200

```
# Executables
Simply run `python MachineLearningMCMC -c /path/to/toml/config` and it'll automatically run the chain. 

# Implementing a New Fitter
Implementing a new fitter is relatively simple. Most implementing is done in `machine_learining/ml_factory/MLFactory`. For Scikit-Learn based models, the new method just needs to imported and added to the `scikit` entry in `__IMPLEMENTED_ALGORITHMS`.

For non-scikit based algorithms currentlt no implementation exists. For such cases a new interface class (which inherits from `FMLInterface`) needs to be implemented. Hopefully in future this is easy to do!

# TODO:
* More libs than scikit
* Configurable fitters
* Clearer Readme