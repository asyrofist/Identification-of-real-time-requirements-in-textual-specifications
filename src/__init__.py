import sys
import os

work_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.extend([work_dir])
print(work_dir)
