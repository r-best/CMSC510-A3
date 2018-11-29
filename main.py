import sys
import numpy as np

from A3 import KLR, SKLR


RUNS = 25

metrics = np.zeros(7)
for _ in range(RUNS):
    np.add(metrics, SKLR.main(sys.argv))
np.divide(metrics, RUNS)

print(metrics)
