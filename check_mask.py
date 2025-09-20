# check_mask.py
import sys, numpy as np
from PIL import Image
m = np.array(Image.open(sys.argv[1]).convert("P"))
u,c = np.unique(m, return_counts=True)
print("labels:", u.tolist(), "counts:", c.tolist(), "nonzero%:", 100*(m!=0).mean())