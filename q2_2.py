'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.

    slides p29
    '''
    eta = np.zeros((10, 64))
    a = b = 2
    for k in range(10):
        k_digits = data.get_digits_by_label(train_data, train_labels, k)
        # n = nh + nt
        n = len(k_digits)
        for j in range(64):
            nh = np.sum(k_digits, axis=0)[j]
            eta[k, j] = (nh + a - 1)/(n + a + b -2)
    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    all = []
    for i in range(10):
        img_i = class_images[i]
        img_i = img_i.reshape((8, 8))
        all.append(img_i)

    all_concat = np.concatenate(all, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    for k in range(10):
        for j in range(64):
            generated_data[k, j] =  np.random.binomial(1, p=eta[k, j])
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    num_data = int(bin_digits.shape[0])
    likelihoods = np.zeros((num_data, 10))
    for i in range(num_data):
        for k in range(10):
            for j in range(64):
                left = eta[k, j]**bin_digits[i, j]
                right = (1 - eta[k, j])**(1 - bin_digits[i, j])
                if likelihoods[i, k] == 0:
                    likelihoods[i, k] = left * right
                else:
                    likelihoods[i, k] *= left * right
    likelihoods = np.log(likelihoods)
    return likelihoods

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class

    p(y|x, eta)=p(x|y, eta)*p(y)/p(x|eta)
    after log:
    log p(y|x, eta)=log p(x|y, eta) + log p(y) - log p(x|eta)

    p(x|eta) =  sum y over p(x, y| eta)
    '''
    log_likelihoods = generative_likelihood(bin_digits, eta)
    likelihoods = np.exp(log_likelihoods)

    prior = 1/10
    log_prior = np.log(prior)


    evi = np.sum(likelihoods, axis=1) * prior
    tmp = []
    for i in evi:
        tmp.append(np.array([i]))
    evi = np.array(tmp)
    log_evi = np.log(evi)

    cond = log_likelihoods + log_prior - log_evi

    return cond

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    n = len(cond_likelihood)
    summation = 0
    for i in range(n):
        label_ind = int(labels[i])
        summation += cond_likelihood[i, label_ind]

    # Compute as described above and return
    return summation / n

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class
    results = []
    for i in range(len(cond_likelihood)):
        results.append(cond_likelihood[i].tolist().index(max(cond_likelihood[i])))
    results = np.array(results)
    return results


def classification_accuracy(classified_data, labels):
    '''
    Calculate the accuracy.
    '''
    total = 0
    n = len(labels)
    for i in range(n):
        if classified_data[i] == labels[i]:
            total += 1
    accuracy = total / n

    return accuracy


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)
    # print(eta)
    generate_new_data(eta)

    avg_train_likeihood = avg_conditional_likelihood(train_data, train_labels, eta)
    print("avg_train_likeihood: ", avg_train_likeihood)
    avg_test_likeihood = avg_conditional_likelihood(test_data, test_labels, eta)
    print("avg_test_likeihood: ", avg_test_likeihood)
    
    train_classified_data = classify_data(train_data, eta)
    train_accuracy = classification_accuracy(train_classified_data, train_labels)
    print("train_accuracy: ", train_accuracy)
    test_classified_data = classify_data(test_data, eta)
    test_accuracy = classification_accuracy(test_classified_data, test_labels)
    print("test_accuracy: ", test_accuracy)


if __name__ == '__main__':
    main()
