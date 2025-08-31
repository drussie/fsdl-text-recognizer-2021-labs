# run_lab3_tf32.py
import sys, importlib
import torch

# Optional performance knobs
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"
torch.set_float32_matmul_precision("high")

# Pass through CLI args to the experiment module
mod = importlib.import_module("lab3.training.run_experiment")
# Make argparse in run_experiment see the same argv
# (first argv entry is normally the script/module name)
sys.argv = ["lab3.training.run_experiment", *sys.argv[1:]]

if __name__ == "__main__":
    mod.main()  # call the module's main()
