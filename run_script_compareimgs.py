from weighted_summary import main
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('inhibit_length', type=int)
parser.add_argument('img_name', type=str)
# parser.add_argument('inhibit_thresh', type=float)
parser.add_argument('inhibit_thresh', type=json.loads)
parser.add_argument('kernel_name', type=str)
args = vars(parser.parse_args())

print(args['inhibit_thresh'])

# no brackets 
#main(inhibit_length=args['inhibit_length'], img_name=args['img_name'], inhibit_thresh=args['inhibit_thresh'], kernel_name=args['kernel_name'],
#        exp_thresholds=[1.0]) # exp_thresholds - often 0.1,1,1.0.  

# brackets 
main(inhibit_length=args['inhibit_length'], img_name=args['img_name'], inhibit_thresh=args['inhibit_thresh'], kernel_name=args['kernel_name'],
        exp_thresholds=[0.01, 0.1, 1.0, 10.0, 100.0]) # exp_thresholds - often 0.1,1,1.0.  
