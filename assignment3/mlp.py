import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learningRate = 0.01
iterations = 10
batchSize = 100
hiddenLayers = [25, 75]
labels = 10
inputPixels = 784
l2Regularization = 0.01


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    for i in hiddenLayers:
        X = tf.placeholder("float", [None, inputPixels])
        Y = tf.placeholder("float", [None, labels])

        weight1 = tf.Variable(tf.random_normal([inputPixels, i]))
        weight2 = tf.Variable(tf.random_normal([i, labels]))
        bias1 = tf.Variable(tf.random_normal([i]))
        bias2 = tf.Variable(tf.random_normal([labels]))

        layer1 = tf.nn.relu(tf.matmul(X, weight1) + bias1)
        layer2 = tf.matmul(layer1, weight2) + bias2

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=layer2, labels=Y) + l2Regularization * tf.nn.l2_loss(weight2))
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer2, labels=Y))
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learningRate).minimize(cost)
        initialize = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(initialize)

            for j in range(iterations):
                meanCost = 0
                xyz = int(mnist.train.num_examples / batchSize)
                for k in range(xyz):
                    batchX, batchY = mnist.train.next_batch(batchSize)
                    newOptimizer, newCost = sess.run(
                        [optimizer, cost], feed_dict={X: batchX, Y: batchY})
                    meanCost += float(newCost / xyz)

            prediction = tf.equal(tf.argmax(layer2, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
            print("Classification Error on Training Images for Hidden layer of size ", i,
                  " is : ", 100 - (accuracy.eval({X: mnist.train.images, Y: mnist.train.labels})) * 100)
            print("Classification Error on Test Images for Hidden layer of size ", i,
                  " is : ", 100 - (accuracy.eval({X: mnist.test.images, Y: mnist.test.labels})) * 100)


if __name__ == '__main__':
    main()
