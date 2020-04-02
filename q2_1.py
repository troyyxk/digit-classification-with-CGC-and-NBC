'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = []
    for i in range(10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        # print(i_digits.shape)
        i_digits_mean = np.mean(i_digits, axis=0)
        means.append(i_digits_mean)

    return np.array(means)

def compute_sigma_mles(train_data, train_labels):
    all = []
    for i in range(10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        all.append(np.cov(i_digits.T))

    return np.array(all)


# def compute_sigma_mles(train_data, train_labels):
#     '''
#     Compute the covariance estimate for each digit class

#     Should return a three dimensional numpy array of shape (10, 64, 64)
#     consisting of a covariance matrix for each digit class 
#     '''


#     covariances = []
    
#     means = compute_mean_mles(train_data, train_labels)
    
#     for i in range(10):
#         i_digits = data.get_digits_by_label(train_data, train_labels, i)
#         i_mean = means[i]
#         diff_matrices = []
#         for j in range(int(i_digits.shape[0])):
#             diff = i_digits[i] - i_mean
#             m_diff = diff[np.newaxis]
#             diff_matrices.append(m_diff.T.dot(m_diff))
#         covariances.append(np.mean(np.array(diff_matrices), axis=0))

#     stabilizer = 0.01*np.identity(64)

#     for i in range(10):
#         covariances[i] = covariances[i]+stabilizer

#     covariances = np.array(covariances)

#     # Compute covariances
#     return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    all_num = []
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        all_num.append(np.log(cov_diag).reshape(8,8))

    all_concat = np.concatenate(all_num, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    num_data = int(digits.shape[0])
    likelihoods = np.zeros((num_data, 10))
    for i in range(num_data):
        for j in range(10):
            left = ((2*np.pi)**(-64/2))*((np.linalg.det(covariances[j]))**(-1/2))
            diffs = digits[i] - means[j]
            inner = np.linalg.inv(covariances[j]).dot(diff)
            right = np.exp((-1/2)*diff.T.dot(inner))
            likelihoods[i,j] = np.log(left * right)
    return likelihoods

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    return None

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    return None

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    pass

def main():
    print("Enter main")
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    # a = tmp(train_data, train_labels)
    # b = np_cov(train_data, train_labels)
    plot_cov_diagonal(covariances)
    
    # Evaluation

if __name__ == '__main__':
    main()