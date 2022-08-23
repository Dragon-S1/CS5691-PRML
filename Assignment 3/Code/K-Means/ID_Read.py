import numpy as np
import os

#reading data from Image Dataset directory
def read_data(dataset, type):
	curr = os.getcwd()
	path = f"../../Assets/K-Means/Image Dataset/{dataset}/{type}"
	os.chdir(path)

	dataList = []
	files = [file for file in os.listdir()]

	for file in files:
		dataList.append(np.genfromtxt(file, delimiter=' ', dtype=float))
	os.chdir(curr)
	return np.array(dataList)