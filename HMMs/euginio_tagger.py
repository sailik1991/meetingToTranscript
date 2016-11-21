import csv
import string
import sys
import cPickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def create_data(file_name):
	'''with open('../../consistency_ami_kim.csv', 'rb') as data_file:'''
	with open(file_name, 'rb') as data_file:
		print 'loading data'
		data_reader = csv.reader(data_file, delimiter = "|")
		headers = data_reader.next()
		data = {}
		for title in headers:
			data[title] = []
		for row in data_reader:
			for header, value in zip(headers, row):
				if header == "words":
					data[header].append((value.translate(None, string.punctuation)).lower())
				else:
					data[header].append(value)
		print 'data created with '+ str(len(data))+' columns and '+ str(len(data['words']))+' rows'
	return data

def create_online_test_data(file_name):
	with open(file_name, 'rb') as data_file:
		print 'loading data'
		data_reader = csv.reader(data_file, delimiter = "|")
		data = {'words': []}
		for row in data_reader:
			data['words'].append((row.translate(None, string.punctuation)).lower())
	return data

'''def convert_euginio_features(data):
	data["ea-tag"] = []
	length = len(data["words"])
	for i in range(length-1, -1, -1):
		if		
'''

def train(data, data_col, target_col):
	print 'start training'
	tv = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
	X = tv.fit_transform(data[data_col])
	with open("feature.pkl", "wb") as fid:
		cPickle.dump(tv, fid)
	mb = MultinomialNB()
	model = mb.fit(X, data[target_col])
	return model

def test(model, data, data_col, target_col):
	print 'start testing'
	#tv = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
	with open("feature.pkl", "rb") as fid:
		tv = cPickle.load(fid)
	X = tv.transform(data[data_col])
	if not target_col:
		return model.predict(X)
	else:
		return model.score(X, data[target_col])

def train_wrapper(args):
	data = create_data(args[2])
	model = train(data, "words", "Da-name")
	if args[3]:
		file_name = args[3] + '.pkl'
		with open(file_name, 'wb') as fid:
			cPickle.dump(model, fid)

def test_wrapper(args):
	data = create_online_test_data(args[2])
	file_name = args[4] + '.pkl'
	with open(file_name, 'rb') as fid:
		model = cPickle.load(fid)
	result = test(model, data, "words", "")
	f = open(args[3], "w")
	f.writelines(result)
	f.close()

def complete_wrapper(args):
	print 'end to end testing started'
	data = create_data(args[2])
	#train = dict(data.items()[len(data)*3/4):])
	#test = dict(data.items()[:len(data)*3/4])
	length = len(data["words"])*4/5
	train_data = {}
	test_data = {}
	for key in data.keys():
		train_data[key] = data[key][:length]
		test_data[key] = data[key][length:]
	model = train(data, "words", "Da-name")
	result = test(model, data, "words", "Da-name")
	print "percentage result on test data " + str(result)

''' Reads arguments from the command line and calls the appropriate function 
	argument 1 - which function should we call values
		train - to train the model using csv file. saves the file for future
		complete - train it and test it using a part of the data
		test - test for online data
	argument 2 - if there is a file that should be provided
		train - this will be the data file
		complete - same data file
		test - speech to text file
	argument 3 - this would be the file name in which the output will be written
		train - output file name
		test - output file name where each line would have the corresponding tag
	argument 4 - if the model has to be picked from a file
		test - model file from where the model should be picked
'''
def main():
	args = sys.argv
	print len(args)
	options = {
		"train": train_wrapper,
		"test": test_wrapper,
		"complete": complete_wrapper,
	}
	options[args[1]](args)

if __name__ == '__main__':
	main()
