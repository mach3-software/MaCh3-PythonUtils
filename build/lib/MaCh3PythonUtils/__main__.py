import argparse
from config_reader import ConfigReader

def main():
    parser = argparse.ArgumentParser(usage="python make_plots -c <config_name>.yaml")
    parser.add_argument("-c", "--config", help="yaml config file", required=True)

    args = parser.parse_args()
    
    config_reader = ConfigReader(args.config)
    config_reader()
    
    

if __name__=="__main__":
    main()