import os
from utils.data_utils import verify_dataset
from utils.data_utils import verify_archive

assert(os.path.isdir(verify_dataset('train2014')))
assert(verify_archive('inpainting.tar.bz2'))

assert(verify_dataset('patate') == None)
assert(verify_archive('patate.tar.bz') == False)