# MaCh3-PythonUtils
[![Release](https://img.shields.io/github/release/mach3-software/MaCh3-PythonUtils.svg)](https://github.com/mach3-software/MaCh3-PythonUtils/releases/latest)
[![Code - Documented](https://img.shields.io/badge/Code-Documented-2ea44f)]([https://github.com/mach3-software/MaCh3/wiki](https://mach3-software.github.io/MaCh3-PythonUtils/))

Very simple tool for analysing MCMC. Currently only accepts chains where all variables are saved within a ROOT TTree. 

## Setup
Setup is relatively simple. The recommended way of running this is using a virtual environment
```bash
virtualenv .env
```

Then to setup,
```bash
source .env/bin/activate
```
The module can then be installed through pip with
```bash
pip install .
```


## Running
Running the package is also simple. pip adds `mach3_python_utils` as an executable so you simply need to run
```
mach3_python_utils -c /path/to/config.yml
```
This command can be accessed anywhere

Some example configs can be found in the `configs` folder.

##  Configs
Configs are in YAML format

For all packages the initial setup is very similar:

```yaml
FileSettings:
    FileName : "/path/to/file"  # Name of File
    ChainName : "/tree/name"    # Name of Tree in file containing chain
    # Do you want it to be verbose?
    Verbose: False
    
    # What plots do we want?
    MakeMLModel: True # ML stuff
    MakeDiagnostics: True # Make MCMC diagnostic plots
    MakePosteriors: True # Make posteriors    

# Settings for parameter options
ParameterSettings:
    # Names of parameters to fit, finds all parameters containing names in this string as sub-string
    ParameterNames : ["sin2th", "sin2th","delm2_12", "delta", "xsec"]
    # Name of variable you're fitting in
    LabelName : "LogL"
    # Parameters you don't want to include in the model
    IgnoredParameters : ["LogL_systematic_xsec_cov", "Log", "LogL_systematic_nddet_cov", ]
    # Any cuts, for MaCh3 I'd recommend capping LogL
    ParameterCuts : ["LogL<12345678", "step>10000"]
    CircularParameters: [] # Do any parameters loop?

```

## Plotting Settings
This package contains various plotting tools required for analysing markov chains which are stored in the Plotting library!

```yaml
PlottingSettings:
  DiagnosticsSettings:
    # Where are we prioting?
    DiagnosticsOutputFile: "diagnostics_output.pdf"
    # Make Trace/AC Plot?
    MakeTraceAC: True
    # Make Violin Plot?
    MakeViolin: False
    #Make ESS Plot?
    MakeESS: False
    # Make MCSE Plot
    MakeMCSE: False
    # Make suboptimality Plot
    MakeSuboptimality: False
    # Steps/calculation for subopt.
    SuboptimalitySteps: 10000
    # Print summary stats
    PrintSummary: True

  PosteriorSettings:
    # Output PDF
    PosteriorOutputFile: "posteriors.pdf"
    # Do you want 1D CIs?
    Make1DPosteriors: True
    # Plotted CIs
    CredibleIntervals: [0.6, 0.90, 0.95]
    # Do you want 2D CIs?
    Make2DPosteriors: False
    # Do you want a triangle plot?
    MakeTrianglePlot: False
    # variables to put in the triangle

```

## ML Settings

For scikit learn based packages the settings are then set in the following way (where FitterKwargs directly sets the keyword arguments for the scikit fitting tool being used):
```yaml
MLSettings:
    # Package model is included in
    FitterPackage : "SciKit" 

    # Fitter Name
    FitterName : "HistBoost"

    # Size of test set (range is 0-1)
    TestSize : 0.8
    # Set fitter Hyper Parameters, these are found in the fitter's readme
    FitterKwargs:
            # n_jobs = 32
            verbose: True
            max_iter: 10000
            # n_estimators = 200
```

For TensorFlow based packages the settings are more complex
```yaml
FitterSettings:

    FitterPackage : "TensorFlow"
    FitterName : "Sequential"
    TestSize : 0.9

    FitterKwargs:
        BuildSettings:
            optimizer: 'adam'
            loss: 'mse'

        FitSettings:
            epochs: 20
            batch_size: 20

        Layers:
            - dense:
                units: 128
            - dense:
                units: 64
            - dropout:
                rate: 0.5
            - dense: 
                units: 16
            - dense:
                units: 1

```


Here FitterKwargs is now split into sub-settings with `BuildSettings` being passed to the model `compile` method, `FitSettings` setting up training information, and `Layers` which defines the types + kwargs of each layer in the model. New layers can be implemented in the `__TF_LAYER_IMPLEMENTATIONS` object which lives in `machine_learning/tf_interface`

## Implementing a New Fitter
Implementing a new fitter is relatively simple. Mostly this is done in `machine_learining/ml_factory/MLFactory`. For Scikit-Learn based models, the new method just needs to imported and added to the `scikit` entry in `__IMPLEMENTED_ALGORITHMS`.

For non-scikit/tf based algorithms currently no implementation exists. For such cases a new interface class (which inherits from `FMLInterface`) needs to be implemented. Hopefully in future this is easy to do!


## TO DO LIST:
- [ ] Better diagnostic plotting (particularly for the NNs)
- [ ] Smart hyper parameter tuning (Just a random grid search will do!)
- [ ] Better sampling methods that "MCMC output go brrrrrr"

