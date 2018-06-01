#!/bin/python

import numpy as np
import nynn

network = nynn.network(3,6,1,1)
myin = np.array([ [1,3,2,42,1,23],
	          [32,2,4,2,11,0],
		  [1,4,5,7,9,12] ])
myout = np.array([ [1],
		   [2],
		   [3] ])
for iter in range(10000):
	out = network.forward(myin)
	err = network.backprop(myout)
print(network.forward(myin))
