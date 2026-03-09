import torch
import transformers
import datasets
import accelerate
import peft
import trl
import bitsandbytes
import numpy



print("🔍 当前环境下库的版本：")
print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"datasets: {datasets.__version__}")
print(f"accelerate: {accelerate.__version__}")
print(f"peft: {peft.__version__}")
print(f"trl: {trl.__version__}")
print(f"bitsandbytes: {bitsandbytes.__version__}")
print(f"numpy: {numpy.__version__}")

