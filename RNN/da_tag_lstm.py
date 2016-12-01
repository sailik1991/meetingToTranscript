import numpy as np
import tensorflow as tf
import subprocess

learning_rate = 0.001
n_classes = 15
n_input = 100#738#313# # size of word embeddings
n_steps = 1 # timesteps
n_hidden = 70#120# # hidden layer num of features
max_epochs = 60

data_len = 495568

#p = subprocess.check_output('wc -l /home/baxter/DL/word2vec_google/591train.we', shell=True)
# http://stackoverflow.com/questions/18739239/python-how-to-get-stdout-after-running-os-system
#print "xxx", int(p[:p.index(' ')])

def extract_data(file_name, label_file):
    data = np.zeros((data_len,n_input), dtype=np.float32)
    seq_len_list = []
    seq_len = 0
    np_label = np.zeros(shape=1)

    with open(file_name, 'rb') as data_file:
        i = 0
        for line in data_file:
            data[i] = np.array(line.split(' ', 1)[1].split(), dtype='float32')
            seq_len += 1
            if line[0] is '.':
                seq_len_list.append(seq_len)
                seq_len = 0
            i += 1
            print i

    with open(label_file, 'rb') as label_file:
        i = 0
        for line in label_file:
            assert seq_len_list[i] is not 0
            a = [np.float32(line)]
            lb = np.lib.pad(a, (0,seq_len_list[i]-1), 'edge')
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
            np_label = np.concatenate((np_label,lb))
            i += 1
    return data, np_label[1:]

#data, target = extract_data('/home/baxter/DL/word2vec_google/591test.we', '/home/baxter/DL/cse591/591test_label.txt')
#assert len(data) == len(target), ('data.shape: %s labels.shape: %s' % (len(data), len(target)))

class DataHandler(object):

    def __init__(self, data_pb):
        self.batch_size_ = 0
        p = subprocess.check_output('wc -l ' + data_pb.data_file, shell=True)
        self.data_length_ = int(p[:p.index(' ')])
        self.n_classes = data_pb.n_classes
        self._seq_len_list = []
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.data_, self.labels_ = self.extract_data(data_pb.data_file, data_pb.labels_file)
        assert self.data_.shape[0] == self.labels_.shape[0], (
            'data.shape: %s labels.shape: %s' % (self.data_.shape,
                                                 self.labels_.shape))

    def extract_data(self, data_file, labels_file):
        data = np.zeros((self.data_length_, n_input), dtype=np.float32)
        seq_len = 0
        np_label = np.zeros(shape=1)

        with open(data_file, 'rb') as data_file:
            i = 0
            for line in data_file:
                data[i] = np.array(line.split(' ', 1)[1].split(), dtype='float32')
                seq_len += 1
                if line[0] is '.':
                    self._seq_len_list.append(seq_len)
                    seq_len = 0
                i += 1

        with open(labels_file, 'rb') as label_file:
            i = 0
            for line in label_file:
                assert self._seq_len_list[i] is not 0
                a = [np.float32(line)]
                lb = np.lib.pad(a, (0, self._seq_len_list[i] - 1), 'edge')
                # https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
                np_label = np.concatenate((np_label, lb))
                i += 1
        print data.shape, np_label.shape[0]-1
        return data, self.dense_to_one_hot(np_label[1:], self.n_classes)

    def dense_to_one_hot(self, labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        T = index_offset + labels_dense.ravel()
        for t in T:
            labels_one_hot.flat[t] = 1
        return labels_one_hot

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch >= self.data_length_:
            # Finished epoch
            self._epochs_completed += 1
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self.data_length_

        end = self._index_in_epoch
        return self.data_[start:end], self.labels_[start:end]

class TrainProto(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.data_file = '/home/baxter/DL/word2vec_google/591train.we'  # FssTrain.txt'#
        self.labels_file = '/home/baxter/DL/cse591/591train_label.txt'  # LssTrain.txt'#

class TestProto(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.data_file = '/home/baxter/DL/word2vec_google/591test.we'  # FssTrain.txt'#
        self.labels_file = '/home/baxter/DL/cse591/591test_label.txt'

# tf Graph input
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, n_input], name='x')
    y = tf.placeholder(tf.float32, [None, n_classes], name='y')

def BiRNN(x):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    with tf.name_scope('weights'):
        # Define weights
        weights = {
            # import osHidden layer weights => 2*n_hidden because of foward + backward cells
            'hidden': tf.Variable(tf.random_normal([n_input, 2 * n_hidden]),name='w_hidden'),
            'out': tf.Variable(tf.random_normal([2 * n_hidden, n_classes]),name='w_out')
        }
    with tf.name_scope('biases'):
        biases = {
            'hidden': tf.Variable(tf.random_normal([2 * n_hidden]),name='b_hidden'),
            'out': tf.Variable(tf.random_normal([n_classes]),name='b_out')
        }

    # Reshape to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
#    x = tf.matmul(x, weights['hidden']) + biases['hidden']
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define lstm cells with tensorflowyoccyoucest_len, han123

    # Forward direction cell
    with tf.variable_scope('forward', reuse=None):
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    with tf.variable_scope('backward', reuse=None):
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    # Only use the output at the last time frame
    # http://stackoverflow.com/questions/36764791/in-tensorflow-how-to-use-tf-gather-for-the-last-dimension

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = BiRNN(x)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1), name='correct_pred')
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    tf.scalar_summary('accuracy',accuracy)

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.merge_all_summaries()
# create saver
saver = tf.train.Saver()
# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    train_writer = tf.train.SummaryWriter('/home/baxter/DL/cse591/meetingToTranscript-master/train', sess.graph)
    test_writer = tf.train.SummaryWriter('/home/baxter/DL/cse591/meetingToTranscript-master/test')

    sess.run(init)
    step = 0
    train_acc = []
    data_pb = TrainProto(n_classes)
    dhTrain = DataHandler(data_pb)
    # Keep training until reach max iterations
    for epoch in xrange(max_epochs):
        while dhTrain._epochs_completed == 0:
            dhTrain._index_in_epoch = sum(dhTrain._seq_len_list[:step])
            batch_x, batch_y = dhTrain.next_batch(dhTrain._seq_len_list[step])
            batch_x = batch_x.reshape((dhTrain._seq_len_list[step], n_input))
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch accuracy
            summary, acc = sess.run([merged,accuracy], feed_dict={x: data, y: target})
            # Calculaextract_datate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            train_writer.add_summary(summary,step)
            step += 1
            if step%10 == 0:
                print "step,epoch", step, epoch
        dhTrain._epochs_completed = 0 # reset epoch completed flag
        step = 0
    print "Optimization Finished!"
    save = saver.save(sess, '/home/baxter/DL/cse591/meetingToTranscript-master/model/591Model')

    # testing
    test_data_pb = TestProto(n_classes)
    dhTest = DataHandler(test_data_pb)
    step = 0
    ACC_ = []
    while dhTest._epochs_completed == 0:
        dhTest._index_in_epoch = sum(dhTest._seq_len_list[:step])
        test_data, test_label = dhTest.next_batch(dhTest._seq_len_list[step])
        summary,acc = sess.run([merged,accuracy], feed_dict={x: test_data, y: test_label})
        ACC_.append(acc)
        test_writer.add_summary(summary)
        step += 1
        if step % 10 == 0:
            print "step", step
    print "Overall accuracy: ", sum(ACC_)/len(ACC_)
