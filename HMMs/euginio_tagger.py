import csv
import string
import sys
import cPickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sequence_generator import SequenceGenerator 
import operator

def create_data(file_name):
	'''with open('../../consistency_ami_kim.csv', 'rb') as data_file:'''
	with open(file_name, 'rb') as data_file:
		print 'loading data'
		data_reader = csv.reader(data_file, delimiter = "|")
		headers = data_reader.next()
		data = {}
		sg = SequenceGenerator()
		topics = sg.getTopicNames()
		data_topics = sg.generateSequences()
		eu_tags = {}
		old_meetingID = ""
		meetingID = ""
		data["eu_tag"] = []
		for topic in topics:
			eu_tags[topic] = convert_topic_das_euginio(data_topics[topic])
		for title in headers:
			data[title] = []
		index = 0
		count = 0
		for row in data_reader:
			# need to find the euginio tags for the group
			for header, value in zip(headers, row):
				if header == "Meeting ID":
					meetingID = value
				if header == "words":
					sentence = value.strip()
					data[header].append((value.translate(None, string.punctuation)).lower())
				else:
					data[header].append(value)
			try:
				if meetingID != old_meetingID:
					meeting_data = get_meeting_transcript(meetingID, topics, data_topics)
					index = 0
				if sentence in meeting_data[index]:
					data["eu_tag"].append(eu_tags[topic][index])
					count += 1
					index += 1
				else:
					data["eu_tag"].append("na")
			except:
				for header in headers:
					data[header] = data[header][:-1]
			old_meetingID = meetingID
		print 'data created with '+ str(len(data))+' columns and '+ str(len(data['words']))+' rows'
		print count
	return data

def get_meeting_transcript(mID, topics, data_topics):
	meeting_topics = [topic for topic in topics if mID in topic]
	start_time = {}
	for topic in meeting_topics:
		temp = data_topics[topic][0].split("|")
		start_time[topic] = float(temp[2])
	sort = sorted(start_time.items(), key = operator.itemgetter(1))
	meeting_topics = []
	meeting_topics.extend(row[0] for row in sort)
	data = []
	for topic in meeting_topics:
		data.extend(data_topics[topic])
	return data

def create_online_test_data(file_name):
	with open(file_name, 'rb') as data_file:
		print 'loading data'
		data_reader = csv.reader(data_file, delimiter = "|")
		data = {}
		data['words'] = []
		for row in data_reader:
			print ''.join(row)
			data['words'].append((''.join(row).translate(None, string.punctuation)).lower())
	return data

def convert_topic_das_euginio(data):
	length = len(data)
	eu_tag = [None] * length
	determinate = True
	for i in range(length-1, -1, -1):
		da = data[i].split("|")
		if determinate and (da[0] == "sug" or "el" in da[0] or da[0] == "off"):
			eu_tag[i] = "prop"
		elif not determinate and (da[0] == "sug" or "el" in da[0] or da[0] == "off"):
			eu_tag[i] = "pdo"
		elif determinate and (da[0] == "off" or (da[0] == "ass" and da[1] == "POS")):
			eu_tag[i] = "commit"
		elif determinate and da[0] == 'inf':
			eu_tag[i] = "uo"
			determinate = False
		else:
			eu_tag[i] = "na"
	return eu_tag

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
	f.writelines("\n".join(result))
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
	model = train(data, "words", "eu_tag")
	result = test(model, data, "words", "eu_tag")
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
