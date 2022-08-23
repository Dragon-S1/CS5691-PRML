# Read train and dev handwritten data

import numpy as np
import os

def read_data(dataset, type, filetype):
	curr = os.getcwd()
	path = f"../../Assets/DTW/Handwritting Data/{dataset}/{type}"
	os.chdir(path)

	dataList = []
	files = [file for file in os.listdir() if file.endswith(filetype)]

	for file in files:
		dataList.append(np.genfromtxt(file, delimiter=' ', dtype=float)[1:].reshape(-1,2))
	os.chdir(curr)
	return dataList