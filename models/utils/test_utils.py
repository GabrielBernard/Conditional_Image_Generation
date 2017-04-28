import os
try:
   from utils.data_utils import verify_dataset
   from utils.data_utils import verify_archive
except ImportError:
    from data_utils import verify_dataset
    from data_utils import verify_archive


assert(os.path.isdir(verify_dataset('train2014')))
assert(verify_archive('inpainting.tar.bz2'))

assert(verify_dataset('patate') is None)
assert(not verify_archive('patate.tar.bz'))

print("All tests passed")
