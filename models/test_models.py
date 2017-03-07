import glob
import os

try:
    from models import data_utils
except ImportError:
    from utils import data_utils

file_path = os.path.split(__file__)[0]
dir = file_path + "/../data/input_test"
input_names = glob.glob(dir + "/input*.jpg")
input_size = (64, 64)

target_names = glob.glob(dir + "/target*.jpg")
target_size = (32, 32)

input_data = data_utils.load_data(input_names, input_size)

assert(input_data.shape == (11, 3, input_size[0], input_size[1]))

target_data = data_utils.load_data(target_names, target_size)

assert(target_data.shape == (11, 3, target_size[0], target_size[1]))

print("All test passed")
