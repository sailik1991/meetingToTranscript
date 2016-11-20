import string
from hmmlearn.hmm import MultinomialHMM
import numpy as np

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
            #print('No value for "topic" when meetingID = {0}; speakerID = {1} and segmentID = {2}'.format(meetingID.strip(), speakerID.strip(), segmentID.strip()))
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
            #print self.segmentData
            self.getSegmentData = False
        try:
            return self.segmentData[ meetingID.strip() + speakerID.strip() + startTime.split('.')[0].strip() ]
        except:
            #print('No value for "segment" when meetingID = {0}; speakerID = {1} and startTime = {2}'.format(meetingID.strip(), speakerID.strip(), startTime.split('.')[0].strip()))
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
                    topicSeq[topicName].append( l[2].strip() )

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
        self.daID = {'ass':0, 'bck':1, 'be.neg':2, 'be.pos':3, 'el.ass':4, 'el.inf':5, 'el.sug':6, 'el.und':7, 'fra':8, 'inf':9, 'off':10, 'oth':11, 'stl':12, 'sug':13, 'und':14}

    def trainHMMs(self, topics, sequences, labels):
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
                #print X1
                if 'weak' in labels[t].lower():
                    X_incon.append( X1 )
                    l_incon.append( len(sequences[t]) )
                else:
                    X_con.append( X1 )
                    l_con.append( len(sequences[t]) )
        
        X1 = [ [self.daID[x.lower().strip()]] for x in self.daID.keys() ]
        X_incon.append(X1)
        l_incon.append( len(X1) )
        X_con.append(X1)
        l_con.append( len(X1) )
        #print np.array(X_con)
        #print np.array(l_con)
        self.con.fit( np.concatenate(X_con), l_con )
        self.incon.fit( np.concatenate(X_incon), l_incon)

    def testHMMs(self, topics, sequences, labels):
        correct = 0
        wrong = 0
        for t in topics:
            try:
                temp = sequences[t]
                temp = labels[t]
            except:
                continue

            if sequences[t]:
                X1 = [[ self.daID[da.lower().strip()] ] for da in sequences[t]]
                c = self.con.score( np.concatenate([X1]), [len(sequences[t])] )
                i = self.incon.score( np.concatenate([X1]), [len(sequences[t])] )
                if c >= i and 'strong' in labels[t].lower():
                    correct += 1
                elif c < i and 'weak' in labels[t].lower():
                    correct += 1
                else:
                    wrong += 1
                    
                #print "Topic {0} :pred: {1} :label: {2}".format(t, our_label, labels[t])
        accuracy = float(correct)/float(correct+wrong)
        print correct
        print str(correct+wrong)
        print accuracy

if __name__ == '__main__':
    sg = SequenceGenerator()
    topics = sg.getTopicNames()
    sequences = sg.generateSequences()
    labels = sg.generateLabels()

    hmml = HMM_Learner(2)
    hmml.trainHMMs(topics, sequences, labels)
    hmml.testHMMs(topics, sequences, labels)
