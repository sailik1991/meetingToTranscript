import string
from hmmlearn.hmm import MultinomialHMM

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
            print('No value for "topic" when meetingID = {0}; speakerID = {1} and segmentID = {2}'.format(meetingID.strip(), speakerID.strip(), segmentID.strip()))
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
            print('No value for "segment" when meetingID = {0}; speakerID = {1} and startTime = {2}'.format(meetingID.strip(), speakerID.strip(), startTime.split('.')[0].strip()))
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
                if 'weak' in labels[t].lower():
                    X_incon.append( sequences[t] )
                    l_incon.append( len(sequences[t]) )
                else:
                    X_con.append( sequences[t] )
                    l_con.append( len(sequences[t]) )
        
        self.con.fit( X_con, l_con )
        self.incon.fit( X_incon, l_incon )

    def testHMMs(self, topics, sequences, labels):
        for t in topics:
            if sequence[t]:
                c = con.predict( sequences[t], len(sequences[t]) )
                i = incon.predict( sequences[t], len(sequences[t]) )
                print "Topic {0} c {1} i {2} label {3}".format(t, c, i, labels[t])

if __name__ == '__main__':
    sg = SequenceGenerator()
    topics = sg.getTopicNames()
    sequences = sg.generateSequences()
    labels = sg.generateLabels()

    hmml = HMM_Learner(2)
    hmml.trainHMMs(topics, sequences, labels)
    hmml.testHMMs(topics, sequences, labels)
