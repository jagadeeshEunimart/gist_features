from GIST import GIST
from tqdm import tqdm
import sys
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import re
import feather
import os
import multiprocessing as mp


param = {
        "orientationsPerScale":np.array([8,8]),
         "numberBlocks":[10,10],
        "fc_prefilt":10,
        "boundaryExtension":32
}
class Dataloader():
	# def __init__(self,input_path):
	# 	self.input_path = input_path
	# 	# self.output_path = output_path
	# 	self.is_dir = 1 if os.path.isdir(input_path) else 0
	# 	print(self.is_dir)

	def get_inputfile(self,input_path):
		if os.path.isdir(input_path):
			# dirctory in images
			path = "{}".format(input_path)
			a = sorted(os.listdir(path))
			file_list = list(map(lambda x: path + x, a))

			return file_list
		else:
			# image file such png, jpg etc..
			path = "{}".format(input_path)
			return [path]
	# def save_feature(self,x):
	# 	if self.is_dir:
	# 		gist_df = pd.DataFrame(x, columns = [f"gist_{i}" for i in range(x.shape[1])])
	# 	else:
	# 		gist_df = pd.DataFrame(x.reshape(1,-1), columns = ["gist_{}".format(i) for i in range(x.shape[1])])

	# 	gist_df.to_feather("./{}".format(self.output_path))	

def _get_gist(param,file_list):
	img_list = list(map(lambda f :np.array(Image.open(f).convert("L")), file_list))
	gist = GIST(param)

	with mp.Pool(mp.cpu_count()) as pool:
		p = pool.imap(gist._gist_extract,img_list[:])
		gist_feature = list(tqdm(p, total = len(img_list)))
	return np.array(gist_feature)


if __name__ == "__main__":

	data = Dataloader()
	# input_path = sys.argv[1]
	base_path = '/home/jagadeesh_vanga_eunimart_com/Amazon_India/'

	categories = []
	for a,b,c in os.walk(base_path):
		categories.append(a.split('/')[-1])
	print(categories)
	categories = categories[1:]
	for category in categories:
		for a,b,c in os.walk(base_path+category):
			for image in c:
				try:
					print(base_path+'/'+category+'/'+image)
					input_path = base_path+category+'/'+image
					file_list = data.get_inputfile(input_path)
					gist_feature = _get_gist(param,file_list)
					print(gist_feature.shape)
				except Exception as e:
					print(e)
	# gist_feature = gist_feature.reshape(-1,1)
	# print(gist_feature.shape)

	# if args.save == True:
		# data.save_feature(gist_feature)
