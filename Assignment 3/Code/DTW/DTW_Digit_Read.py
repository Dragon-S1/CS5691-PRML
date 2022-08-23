# Code to read training and Development data

import numpy as np
import os

def read_data(dataset, type, filetype):
	curr = os.getcwd()
	path = f"../../Assets/DTW/Isolated Digits/{dataset}/{type}"
	os.chdir(path)

	dataList = []
	files = [file for file in os.listdir() if file.endswith(filetype)]

	for file in files:
		dataList.append(np.genfromtxt(file, delimiter=' ', skip_header=1, dtype=float))
	os.chdir(curr)
	return dataList