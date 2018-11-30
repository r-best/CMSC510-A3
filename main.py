import sys
import numpy as np

from A3 import KLR, SKLR


RUNS = 20

metrics = np.zeros(7)
for _ in range(RUNS):
    metrics = np.add(metrics, SKLR.main(sys.argv))
metrics = np.divide(metrics, RUNS)

print(metrics)
