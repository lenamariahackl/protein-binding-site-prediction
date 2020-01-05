from pathlib import Path
import numpy as np
import logging
import shutil
import csv
import random
import collections

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# calculate average number of amino acids, percentage of metal proteins, percentage of small proteins and percentage of nucleus proteins
def calculate_set_info(set, proteins):
	dist_len = 0
	metal = 0
	small = 0
	nucleus = 0
	for protein_id in set:
		dist_len += proteins[protein_id][0]
		metal += proteins[protein_id][1]
		small += proteins[protein_id][2]
		nucleus += proteins[protein_id][3]
	avg_dist_len = dist_len / len(set)
	avg_metal = metal / len(set)
	avg_small = small / len(set)
	avg_nucleus = nucleus / len(set)
	return avg_dist_len, avg_metal, avg_small, avg_nucleus

# split data in train, valid, test set while keeping in mind even length distribution and even class distribution
# input:
#     test_part: percentage of test samples
#     valid_part: percentage of valid samples
#     x_source_path: location of input protein files
#     y_source_path: location of input targets.csv with format: ID,metal,small,nucleus
#     x_target_path: location of output protein files
#     y_target_path: location of output y_train.csv, y_valid.csv, y_test.csv
def split_data(test_part, valid_part, x_source_path, y_source_path = '', x_target_path = 'data/', y_target_path = 'data/'):
	targets = collections.OrderedDict()
	with open('targets.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				line_count += 1
			else:
				targets[row[0]] = (int(row[1]), int(row[2]), int(row[3]))

	proteins = {}
	for file in x_source_path.iterdir():
		if file.name.endswith("distance.npy"):
			protein_id = str(file.name)[:-16]
			proteins[protein_id] = [len(np.load(file)), targets[protein_id][0], targets[protein_id][1],
			                        targets[protein_id][2]]

	test_split = int(np.floor(len(proteins) * test_part))
	valid_split = int(np.floor(len(proteins) * valid_part)) + test_split

	try_nr = 0
	evenly_distributed = False
	while not evenly_distributed:
		keys = list(proteins.keys())
		random.shuffle(keys)
		test = collections.OrderedDict()
		valid = collections.OrderedDict()
		train = collections.OrderedDict()
		for k in keys[:test_split]:
			test[k] = proteins[k]
		for k in keys[test_split:valid_split]:
			valid[k] = proteins[k]
		for k in keys[valid_split:]:
			train[k] = proteins[k]
		test_avg_dist_len, test_avg_metal, test_avg_small, test_avg_nucleus = calculate_set_info(test, proteins)
		valid_avg_dist_len, valid_avg_metal, valid_avg_small, valid_avg_nucleus = calculate_set_info(valid, proteins)
		train_avg_dist_len, train_avg_metal, train_avg_small, train_avg_nucleus = calculate_set_info(train, proteins)

		tresh_aa = 3
		tresh_distr = 0.03
		print(abs(test_avg_dist_len - train_avg_dist_len), abs(test_avg_dist_len - valid_avg_dist_len))
		print(abs(test_avg_metal - train_avg_metal), abs(test_avg_metal - valid_avg_metal))
		print(abs(test_avg_small - train_avg_small), abs(test_avg_small - valid_avg_small))
		print(abs(test_avg_nucleus - train_avg_nucleus), abs(test_avg_nucleus - valid_avg_nucleus))
		similar_number_aa = abs(test_avg_dist_len - train_avg_dist_len) < tresh_aa and abs(test_avg_dist_len - valid_avg_dist_len) < tresh_aa
		similar_class_distr = abs(test_avg_metal - train_avg_metal) < tresh_distr and abs(
			test_avg_small - train_avg_small) < tresh_distr and abs(test_avg_nucleus - train_avg_nucleus) < tresh_distr and abs(test_avg_metal - valid_avg_metal) < tresh_distr and abs(
			test_avg_small - valid_avg_small) < tresh_distr and abs(test_avg_nucleus - valid_avg_nucleus) < tresh_distr

		evenly_distributed = similar_number_aa and similar_class_distr
		if not evenly_distributed:
			try_nr += 1
			print('Not evenly distributed. Try number', try_nr)

	test_nr = 0
	valid_nr = 0
	train_nr = 0
	targets_test = []
	targets_valid = []
	targets_train = []
	for protein_id in proteins:
		if protein_id in test:
			targets_test.append(
				[protein_id, str(targets[protein_id][0]), str(targets[protein_id][1]), str(targets[protein_id][2])])
		elif protein_id in valid:
			targets_valid.append(
				[protein_id, str(targets[protein_id][0]), str(targets[protein_id][1]), str(targets[protein_id][2])])
		elif protein_id in train:
			targets_train.append(
				[protein_id, str(targets[protein_id][0]), str(targets[protein_id][1]), str(targets[protein_id][2])])
		else:
			print('problem: unmatched protein_id ', protein_id)
		for file in x_source_path.iterdir():
			if protein_id in file.name:
				if protein_id in test:
					dir = x_target_path + "test/"
					Path(dir).mkdir(parents=True, exist_ok=True)
					shutil.move(file, dir + str(file.name))
					test_nr += 1
				elif protein_id in valid:
					dir = x_target_path + "valid/"
					Path(dir).mkdir(parents=True, exist_ok=True)
					shutil.move(file, dir + str(file.name))
					valid_nr += 1
				elif protein_id in train:
					dir = x_target_path + "train/"
					Path(dir).mkdir(parents=True, exist_ok=True)
					shutil.move(file, dir + str(file.name))
					train_nr += 1
				else:
					print('problem: unmatched protein_id ', protein_id)
	wtr = csv.writer(open(y_target_path + 'y_test.csv', 'w'), delimiter=',', lineterminator='\n')
	for x in targets_test:
		wtr.writerow(x)
	wtr = csv.writer(open(y_target_path + 'y_valid.csv', 'w'), delimiter=',', lineterminator='\n')
	for x in targets_valid:
		wtr.writerow(x)
	wtr = csv.writer(open(y_target_path + 'y_train.csv', 'w'), delimiter=',', lineterminator='\n')
	for x in targets_train:
		wtr.writerow(x)
	print('The data was successfully split into a test set of', test_nr, 'items, a validation set of', valid_nr,'items and a training set of', train_nr, 'items.')
# logger.info('The data was sucessfully split into a test set of '+test_nr+' items and a train set of '+train_nr+' items.' )

if __name__ == "__main__":
	test_part = 0.1  # percentage of test set samples eg 0.1 = 10%
	valid_part = 0.1  # percentage of validation set samples
	x_source_path = Path('inputs')
	split_data(test_part, valid_part, x_source_path)
