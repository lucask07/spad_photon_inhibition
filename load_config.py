import yaml
import os
import sys

try:
    config = yaml.safe_load(open(os.path.join(os.path.expanduser('~'), '.bernoulli_inhibit/config.yaml')))
except:
    print('Error yaml config file is not configured at ~/.bernoulli_inhibit/config.yaml')
    sys.exit() 
