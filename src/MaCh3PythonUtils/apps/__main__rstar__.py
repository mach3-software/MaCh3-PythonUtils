import argparse
from MaCh3PythonUtils.config_reader.config_reader import ConfigReader  
from MaCh3PythonUtils.diagnostics.rstar import RStar

def main() -> None:
    parser = argparse.ArgumentParser(usage="python make_plots -c <config_name>.yaml")
    parser.add_argument("-c", "--config", help="yaml config file", required=True)
    
    parser.add_argument("-n", "--n_iterations", type=int, default=1, help="Number of iterations for R* diagnostics")
    parser.add_argument("-f", "--files", nargs='+', required=True, help="List of chain files to use for R* diagnostics")

    args = parser.parse_args()

    config_reader = ConfigReader(args.config)
    
    rstar = RStar(args.files, n_fitters=args.n_iterations, **config_reader.get_settings()['MLSettings']['FitterKwargs'])
    rstar.train_models()
    rstar()
    