import argparse

def main()->None:
    from mach3pythonutils.config_reader.config_reader import ConfigReader
    parser = argparse.ArgumentParser(usage="python make_plots -c <config_name>.yaml")
    parser.add_argument("-c", "--config", help="yaml config file", required=True)

    args = parser.parse_args()

    config_reader = ConfigReader(args.config)
    config_reader()
