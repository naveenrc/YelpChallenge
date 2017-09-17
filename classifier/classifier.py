from tensorflow.python.framework import ops
import tensorflow as tf
from utilities import model as md
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import os
import time
import cv2


def model(photos_train, Y_train, photos_test, Y_test, learning_rate=0.0005,
          num_epochs=1500, minibatch_size=128, print_cost=True):
    """
    Implements a three-layer tensorflow neural network:
    LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set
    Y_train -- training set labels
    X_test -- test set
    Y_test -- test set labels
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    im_size = 64
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (n_x, m) = (im_size * im_size * 3, len(photos_train))  # (n_x: input size, m : number of examples in the train set)
    n_y = 6  # n_y : output size
    costs = []  # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    X, Y = md.create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = md.initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = md.forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = md.compute_cost(Z3, Y)

    # Backpropagation: Define the tensorflow optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)
        path = os.path.dirname(__file__)
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1

            minibatches = md.random_mini_batches_chunk(photos_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                x_train_temp = np.array([cv2.imread(os.path.join(path, '../yelpData/resized64/') + i + '.png')
                                        .reshape((1, im_size * im_size * 3)).T
                                         for i in minibatch_X['photo_id']]).T[0]

                # Flatten the training and test images
                x_train_flatten = x_train_temp
                # Normalize image vectors
                x_train = x_train_flatten / 255.
                # Convert training and test labels to one hot matrices
                y_train = md.convert_to_one_hot(minibatch_Y.T, 6)
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost",
                # the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: x_train, Y: y_train})

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        length = int(len(photos_train))
        Y_train = md.convert_to_one_hot(Y_train.T, 6)

        acc = 0
        for i in range(0, 3):
            X_train_temp = np.array([cv2.imread(os.path.join(path, '../yelpData/resized64/') + i + '.png')
                                 .reshape((1, im_size * im_size * 3)).T
                                  for i in photos_train['photo_id']
                                     [int(i * length / 3):int((i+1) * length / 3)]]).T[0]
            accuracy_temp = accuracy.eval(
                {X: X_train_temp / 255, Y: Y_train[:, int(i * length / 3):int((i+1) * length / 3)]})
            acc += accuracy_temp

        print("Train Accuracy: ", acc / 3)
        X_test = np.array([cv2.imread(os.path.join(path, '../yelpData/resized64/') + i + '.png')
                          .reshape((1, im_size * im_size * 3)).T
                           for i in photos_test['photo_id']]).T[0]

        X_test = X_test / 255.
        Y_test = md.convert_to_one_hot(Y_test.T, 6)
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


if __name__ == '__main__':
    x, y = md.get_data_chunk()
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=.05,
                                                        random_state=42,
                                                        shuffle=True, stratify=y)

    start = time.time()
    parameters = model(x_train, y_train, x_test, y_test, num_epochs=500, learning_rate=0.001)
    stop = time.time()
    print("Time elapsed: " + str(stop - start))
    path = os.path.dirname(__file__)
    param_path = os.path.join(path, '../yelpData/') + 'parameters_5.npy'
    np.save(param_path, parameters)
