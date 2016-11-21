import sys
import string
from hmmlearn.hmm import MultinomialHMM
import numpy as np
import random
import pickle

class SequenceGenerator:
	
	def __init__(self):
		self.getTopicData = True
		self.topicData = {}

		self.getSegmentData = True
		self.segmentData ={}

	def getTopicID(self, meetingID, speakerID, segmentID):
		if self.getTopicData == True:
			f = open('topic_data.csv', 'rb')
			for line in f:
				l = line.strip().split('|')
				self.topicData[ l[0].strip() + l[2].strip() + l[3].strip() ] = l[1].strip()
			f.close()
			self.getTopicData = False;
		try:
			return self.topicData[ meetingID.strip() + speakerID.strip() + segmentID.strip() ]
		except:
			return None

	def getTopicNames(self):
		f = open('topic_data.csv', 'rb') 
		topicNames = [] 
		for l in f:     
			topicNames.append( l.split('|')[1].strip() )
		f.close()
		return list(set(topicNames))

	def getSegmentID(self, meetingID, speakerID, startTime):
		if self.getSegmentData == True:
			f = open('segment_ids.csv', 'rb') 
			for line in f:     
				l = line.strip().split('|')
				self.segmentData[ l[0].strip() + l[1].strip() +l[3].split('.')[0].strip() ] = l[2].strip()
			f.close()
			self.getSegmentData = False
		try:
			return self.segmentData[ meetingID.strip() + speakerID.strip() + startTime.split('.')[0].strip() ]
		except:
			return None

	def generateSequences(self):
		topicNames = self.getTopicNames()
		topicSeq = {}
		for t in topicNames:
			topicSeq[t] = []

		f = open('consistency_ami.csv', 'rb')
		for line in f:
			l = line.strip().split('|')
			meetingID = l[0]
			speakerID = l[1]
			startTime = l[4]
			segmentID = self.getSegmentID(meetingID, speakerID, startTime)
			if segmentID != None:
				topicName = self.getTopicID(meetingID, speakerID, segmentID)
				if topicName != None:
					topicSeq[topicName].append( l[6].strip() )

		return topicSeq

	def generateLabels(self):
		labels = {}
		f = open('consistency_labels.csv', 'rb')
		for l in f:
			tnl = l.strip().split('|')
			labels[ tnl[0] ] = tnl[1]
		return labels

class HMM_Learner:
	
	def __init__(self, M):
		self.con = MultinomialHMM ( n_components = M )
		self.incon = MultinomialHMM (n_components = M )
		self.daID = {'ass':0, 'bck':1, 'be.neg':2, 'be.pos':3, 'el.ass':4, 'el.inf':5,
					'el.sug':6, 'el.und':7, 'fra':8, 'inf':9, 'off':10, 'oth':11, 'stl':12,
					'sug':13, 'und':14}

	def addRandomAllSequence(self, X, lengths):
		da_keys = self.daID.keys()
		random.shuffle(da_keys)
		X1 = [ [self.daID[x.lower().strip()]] for x in da_keys ]
		X.append(X1)
		lengths.append( len(X1) )

	def trainHMMs(self, topics, sequences, labels):
		try:
			self.con = pickle.load( open('HMM_consistent.model','rb') )
			self.incon = pickle.load( open('HMM_inconsistet.model','rb') )
		except:
			X_con = []
			l_con = []
			X_incon = []
			l_incon = []
			for t in topics:
				try:
					temp = sequences[t]
					temp = labels[t]
				except:
					continue

				if sequences[t]:
					X1 = [[ self.daID[da.lower().strip()] ] for da in sequences[t]]
					if 'weak' in labels[t].lower():
						X_incon.append( X1 )
						l_incon.append( len(sequences[t]) )
					else:
						X_con.append( X1 )
						l_con.append( len(sequences[t]) )
		   
			# Add two complete random sequence to support Multinomial in HMMs 
			self.addRandomAllSequence(X_incon, l_incon)
			self.addRandomAllSequence(X_con, l_con)
			
			self.con.fit( np.concatenate(X_con), l_con )
			self.incon.fit( np.concatenate(X_incon), l_incon)

			pickle.dump(self.con, open('HMM_consistent.model','wb'))
			pickle.dump(self.incon, open('HMM_inconsistet.model','wb'))

	def testHMMs(self, topics, sequences, labels):
		correct = 0
		wrong = 0
		for t in topics:
			try:
				temp = sequences[t]
				#temp = labels[t]
			except:
				continue

			if sequences[t]:
				X1 = [[ self.daID[da.lower().strip()] ] for da in sequences[t]]
				c = self.con.score( np.concatenate([X1]), [len(sequences[t])] )
				i = self.incon.score( np.concatenate([X1]), [len(sequences[t])] )
				if c >= i:# and 'strong' in labels[t].lower():
					#print "Topics {0}, Labels {1}".format(t, labels[t])
					#correct += 1
					print "Consistent"
				elif c < i:# and 'weak' in labels[t].lower():
					#correct += 1
					#print "Topics {0}, Labels {1}".format(t, labels[t])
					print "Inconsistent"
				else:
					wrong += 1
					
		#accuracy = float(correct)/float(correct+wrong)
		#print accuracy

if __name__ == '__main__':
	sg = SequenceGenerator()
	topics = sg.getTopicNames()
	sequences = sg.generateSequences()
	labels = sg.generateLabels()
	'''
	for t in topics:
		f = open("meet/"+t+".txt", "wb")
		for line in sequences[t]:
			f.write(line + "\n")
		f.close()
	
	for t in topics:
		try:
			labels[t]
		except:
			continue
		f = open("labels/"+t+".txt", "wb")
		f.write(labels[t] + "\n")
        f.close()	
	'''
	hmml = HMM_Learner(2)
	hmml.trainHMMs(topics, sequences, labels)
	topics = ['RemoteFunctionality']
	f = open(sys.argv[1], 'rb')
	test_sequences = { topics[0]: [l.strip() for l in f] }
	print test_sequences
	
	hmml.testHMMs(topics, test_sequences, [])
