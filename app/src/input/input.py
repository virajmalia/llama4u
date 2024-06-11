import importlib.metadata
import argparse

def parse_arguments():
    """ parse input arguments """
    version = importlib.metadata.version('Llama4U')
    parser = argparse.ArgumentParser(description=f'Llama4U v{version}')
    parser.add_argument('-q', '--query', type=str, required=False, help='Single Query')
    parser.add_argument('-v', '--verbose', type=int, required=False, help='Enable verbose output')
    return parser.parse_args()
