import os
import argparse

os.environ['MKL_THREADING_LAYER'] = 'GNU'
parser = argparse.ArgumentParser(description="AML Generic Launcher")
parser.add_argument("--code_dir", default="", 
                        help="The absolute directory to the code.")
parser.add_argument("--num_gpus", default=8, 
                        type=int, help="The number of gpus for training.")  
args, _ = parser.parse_known_args()

os.system("cd {} && bash exp.sh configs/segmenter/segmenter-ori_vit-b16_512x512_160k_ade20k.py --auto-resume".format(args.code_dir))