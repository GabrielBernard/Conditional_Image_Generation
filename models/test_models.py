import glob
import os
import _pickle as pickle

try:
    from models import data_utils
except ImportError:
    from utils import data_utils
#
# file_path = os.path.split(__file__)[0]
# dir = file_path + "/../data/input_test"
# input_names = glob.glob(dir + "/input*.jpg")
# input_size = (64, 64)
#
# target_names = glob.glob(dir + "/target*.jpg")
# target_size = (32, 32)
#
# input_data = data_utils.load_data(input_names, input_size)
#
# assert(input_data.shape == (11, 3, input_size[0], input_size[1]))
#
# target_data = data_utils.load_data(target_names, target_size)
#
# assert(target_data.shape == (11, 3, target_size[0], target_size[1]))
dic = pickle.load(open("/Users/Gabriel/PycharmProjects/Conditional_Image_Generation/data/save_test/data.pkl", 'rb'))
prefixes = ['input_', 'img_']
for i in data_utils.minibatch_dic_iterator(dic=dic, batch_size=100, prefixes=prefixes, data_path="/Users/Gabriel/PycharmProjects/Conditional_Image_Generation/data/save_test/"):
    print(i)

print("All test passed")
