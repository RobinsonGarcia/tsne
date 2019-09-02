
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg') 
import seaborn as sns
from matplotlib.animation import ArtistAnimation


import os

MACHINE_EPSILON = np.finfo(np.double).eps

#adapted from  https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/

#https://towardsdatascience.com/implementing-t-sne-in-tensorflow-manual-back-prop-in-tf-b08c21ee8e06
#https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/
#https://towardsdatascience.com/t-sne-python-example-1ded9953f26
#https://towardsdatascience.com/kl-divergence-python-example-b87069e4b810


# SNE
#
# \begin{equation}
# q_{j|i}=\frac{exp(-||y_i-y_j||^2)}{\sum_{kdifi}exp(-||y_i-y_j||^2)}
# \end{equation}
#
# \begin{equation}
# C = \sum_{i}KL(P_i||Q_i)=\sum_i\sum_jp_{j|i}log\frac{p_{j|i}}{q_{j|i}}
# \end{equation}
#
# Symmetric SNE
#
# \begin{equation}
# q_{ij}=\frac{exp(-||y_i-y_j||^2)}{\sum_{kdifl}exp(-||y_k-y_l||^2)}
# \end{equation}

# # SNE

# In[2]:


def neg_squared_euc_dists(X):
    """Compute matrix containing negative squared euclidean
    distance for all pairs of points in input matrix X

    # Arguments:
        X: matrix of size NxD
    # Returns:
        NxN matrix D, with entry D_ij = negative squared
        euclidean distance between rows X_i and X_j
    """
    # Math? See https://stackoverflow.com/questions/37009647
    sum_X = tf.reduce_sum(X**2, 1)
    D = tf.transpose(-2 * tf.matmul(X, tf.transpose(X)) + sum_X) + sum_X
    return -D

def softmax(X, diag_zero=True):
    """Take softmax of each row of matrix X."""

    # We usually want diagonal probailities to be 0.
    if diag_zero:
        X = tf.matrix_set_diag(X,tf.zeros([X.shape[0].value],dtype=tf.float64))

    return tf.nn.softmax(X)

def calc_prob_matrix(distances, sigmas=None):
    """Convert a distances matrix to a matrix of probabilities."""
    if sigmas is not None:
        two_sig_sq = 2. * tf.reshape(sigmas,(-1, 1))**2
        return softmax(distances / two_sig_sq)
    else:
        return softmax(distances)


# # Symmetric SNE

# \begin{equation}
# p_{ij} = \frac{P + P^T}{2N}
# \end{equation}
#
# \begin{equation}
# q_{ij}=\frac{exp(-||y_i-y_j||^2)}{\sum_{kdifl}exp(-||y_k-y_l||^2)}
# \end{equation}

# In[3]:


def binary_search(eval_fn, target, tol=1e-10, max_iter=10000,
                  lower=1e-20, upper=1000.):
    """Perform a binary search over input values to eval_fn.

    # Arguments
        eval_fn: Function that we are optimising over.
        target: Target value we want the function to output.
        tol: Float, once our guess is this close to target, stop.
        max_iter: Integer, maximum num. iterations to search for.
        lower: Float, lower bound of search range.
        upper: Float, upper bound of search range.
    # Returns:
        Float, best input value to function found during search.
    """
    for i in range(max_iter):
        guess = (lower + upper) / 2.
        val = eval_fn(guess)
        if val > target:
            upper = guess
        else:
            lower = guess
        if np.abs(val - target) <= tol:
            break
    return guess

def calc_perplexity(prob_matrix):
    """Calculate the perplexity of each row
    of a matrix of probabilities."""
    entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
    perplexity = 2 ** entropy
    return perplexity


def perplexity(distances, sigmas):
    """Wrapper function for quick calculation of
    perplexity over a distance matrix."""
    return calc_perplexity(numpy_calc_prob_matrix(distances, sigmas))


def find_optimal_sigmas(distances, target_perplexity):
    """For each row of distances matrix, find sigma that results
    in target perplexity for that role."""
    sigmas = []
    # For each row of the matrix (each point in our dataset)
    for i in range(distances.shape[0]):
        # Make fn that returns perplexity of this row given sigma
        eval_fn = lambda sigma:             perplexity(distances[i:i+1, :], np.array(sigma))
        # Binary search over sigmas to achieve target perplexity
        correct_sigma = binary_search(eval_fn, target_perplexity)
        # Append the resulting sigma to our output array
        sigmas.append(correct_sigma)
    return np.array(sigmas)


# In[4]:


def numpy_neg_squared_euc_dists(X):
    """Compute matrix containing negative squared euclidean
    distance for all pairs of points in input matrix X

    # Arguments:
        X: matrix of size NxD
    # Returns:
        NxN matrix D, with entry D_ij = negative squared
        euclidean distance between rows X_i and X_j
    """
    # Math? See https://stackoverflow.com/questions/37009647
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    return -D

def numpy_softmax(X, diag_zero=True):
    """Take softmax of each row of matrix X."""

    # Subtract max for numerical stability
    e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))

    # We usually want diagonal probailities to be 0.
    if diag_zero:
        np.fill_diagonal(e_x, 0.)

    # Add a tiny constant for stability of log we take later
    e_x = e_x + 1e-8  # numerical stability

    return e_x / e_x.sum(axis=1).reshape([-1, 1])

def numpy_calc_prob_matrix(distances, sigmas=None):
    """Convert a distances matrix to a matrix of probabilities."""
    if sigmas is not None:
        two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
        return numpy_softmax(distances / two_sig_sq)
    else:
        return numpy_softmax(distances)

def numpy_neg_squared_euc_dists(X):
    """Compute matrix containing negative squared euclidean
    distance for all pairs of points in input matrix X

    # Arguments:
        X: matrix of size NxD
    # Returns:
        NxN matrix D, with entry D_ij = negative squared
        euclidean distance between rows X_i and X_j
    """
    # Math? See https://stackoverflow.com/questions/37009647
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    return -D


def numpy_calc_prob_matrix(distances, sigmas=None):
    """Convert a distances matrix to a matrix of probabilities."""
    if sigmas is not None:
        two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
        return numpy_softmax(distances / two_sig_sq)
    else:
        return numpy_softmax(distances)


def numpy_p_conditional_to_joint(P):
    """Given conditional probabilities matrix P, return
    approximation of joint distribution probabilities."""
    return (P + P.T) / (2. * P.shape[0])


# In[5]:


def q_joint(Y):
    """Given low-dimensional representations Y, compute
    matrix of joint probabilities with entries q_ij."""
    # Get the distances from every point to every other
    distances = neg_squared_euc_dists(Y)
    # Take the elementwise exponent
    exp_distances = tf.math.exp(distances)
    # Fill diagonal with zeroes so q_ii = 0
    exp_distances = tf.matrix_set_diag(exp_distances,tf.zeros([exp_distances.shape[0].value],dtype=tf.float64))

    # Divide by the sum of the entire exponentiated matrix
    return exp_distances / tf.reduce_sum(exp_distances), None

def p_conditional_to_joint(P):
    """Given conditional probabilities matrix P, return
    approximation of joint distribution probabilities."""
    return (P + tf.transpose(P)) / (2. * P.shape[0].value)

def p_joint(X, target_perplexity):
    """Given a data matrix X, gives joint probabilities matrix.

    # Arguments
        X: Input data matrix.
    # Returns:
        P: Matrix with entries p_ij = joint probabilities.
    """
    # Get the negative euclidian distances matrix for our data
    distances = numpy_neg_squared_euc_dists(X)
    # Find optimal sigma for each row of this distances matrix
    sigmas = find_optimal_sigmas(distances, target_perplexity)
    # Calculate the probabilities based on these optimal sigmas
    p_conditional = numpy_calc_prob_matrix(distances, sigmas)
    # Go from conditional to joint probabilities matrix
    P = numpy_p_conditional_to_joint(p_conditional)
    return P



def q_tsne(Y):
    """t-SNE: Given low-dimensional representations Y, compute
    matrix of joint probabilities with entries q_ij."""
    distances = neg_squared_euc_dists(Y)
    inv_distances = (1. - distances)**(-1)
    inv_distances = tf.matrix_set_diag(inv_distances,tf.zeros([inv_distances.shape[0].value],dtype=tf.float64))
    return inv_distances / tf.reduce_sum(inv_distances), inv_distances


# In[6]:


def tsne_grad(P, Q, Y, inv_distances):
    """Estimate the gradient of t-SNE cost with respect to Y."""
    pq_diff = P - Q
    pq_expanded = tf.expand_dims(pq_diff, 2)
    y_diffs = tf.expand_dims(Y, 1) - tf.expand_dims(Y, 0)

    # Expand our inv_distances matrix so can multiply by y_diffs
    distances_expanded = tf.expand_dims(inv_distances, 2)

    # Multiply this by inverse distances matrix
    y_diffs_wt = y_diffs * distances_expanded

    # Multiply then sum over j's
    grad = 4. * tf.reduce_sum(pq_expanded * y_diffs_wt,1)
    return grad

def backprop(C,P,Q,Y,Inv_distances):
    grad = tf.gradients(C,Y)#grad_fn(P, Q, Y, inv_distances)#
    update_Y = []
    update_Y.append(tf.assign(m,m*beta1 + (1-beta1)*(grad)))
    update_Y.append(tf.assign(v,v*beta1 + (1-beta1)*(grad**2)))
    m_hat = m/(1-beta1)
    v_hat = v/(1-beta2)
    adam = lr/(tf.sqrt(v_hat)+adam_e)
    update_Y.append(tf.assign(Y,Y-adam*m_hat))
    return grad,update_Y
# In[7]:


def parse_function(filename, label):
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize_images(image, [224, 224])

    return tf.squeeze(tf.reshape(image,[-1,224*224*3])), label




def get_dataset(root):
    classes = os.listdir(root)
    class_dict = {k:v for v,k in enumerate(classes)}
    print(class_dict)
    filenames = []
    labels = []
    for i in classes:
        path = os.path.join(root,i)
        files = os.listdir(path)
        for ii in files:
            filenames.append(os.path.join(path,ii))
            labels.append(class_dict[i])
            #print(path,i,class_dict[i])

    batch_size = len(filenames)
    dataset = tf.data.Dataset.from_tensor_slices((np.array(filenames),                                                  np.array(labels)))
    dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(parse_function, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    init_op = iterator.initializer
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(1):
            X,y = sess.run(next_element)
            print(X.shape)
            X_shape = X.shape
        sess.run(init_op)
    return dataset,classes,next_element,init_op,X_shape




import os
from PIL import Image
"""
root ='tsne/images'
images = os.listdir(root)
X = []
for i in images:
    img = os.path.join(root,i)
    img = Image.open(img).resize((200,200))
    img = np.array(img).flatten()
    X.append(img)
X = np.vstack(X)
"""

def update(i):
    label = 'timestep {0}'.format(i)
    print(label)
    sc.set_offsets(np.c_[b[-1][:,0]                                 ,b[-1][:,1]])
    return sc

from sklearn.datasets import load_digits
#X, y = load_digits(return_X_y=True)



# In[ ]:
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_dimensions', 2, 'Number of dimensions in P_i.')
flags.DEFINE_integer('perplexity', 50, 'Perplexity.')
flags.DEFINE_float('learning_rate', .1,
                   'Learning rate.')
flags.DEFINE_string('root', 'dataset/train',
                    'Path to the folder containing the image data separated by classes in subfolders.')



if __name__ == "__main__":
    root = FLAGS.root
    dataset,classes,next_element,init_op,X_shape = get_dataset(root)

    q_fn = q_tsne
    grad_fn = tsne_grad
    N,D = X_shape
    d = FLAGS.num_dimensions
    PERPLEXITY = FLAGS.perplexity

    beta1 = 0.9
    beta2 = 0.99
    adam_e = MACHINE_EPSILON
    lr = FLAGS.learning_rate
    grad_fn = tsne_grad

    P = tf.placeholder(tf.float64,shape=[N,N])

    Y = tf.get_variable("Y",shape=[N,d],dtype=tf.float64,initializer=tf.random_normal_initializer())

    m,v = tf.Variable(tf.zeros_like(Y)),tf.Variable(tf.zeros_like(Y))

    Q, inv_distances = q_fn(Y)

    Q= tf.maximum(Q,MACHINE_EPSILON)

    C = tf.reduce_sum( tf.reshape(P,[-1]) * tf.log(tf.maximum(tf.reshape(P,[-1]),MACHINE_EPSILON)                                               / tf.reshape(Q,[-1])))#(tf.reshape(Q,[-1])+tf.constant(1.e-10,dtype=tf.float64))))

    grad = grad_fn(P, Q, Y, inv_distances)
    update_Y = []
    update_Y.append(tf.assign(m,m*beta1 + (1-beta1)*(grad)))
    update_Y.append(tf.assign(v,v*beta1 + (1-beta1)*(grad**2)))
    m_hat = m/(1-beta1)
    v_hat = v/(1-beta2)
    adam = lr/(tf.sqrt(v_hat)+adam_e)
    update_Y.append(tf.assign(Y,Y-adam*m_hat))


    with tf.Session() as sess:
        fig = plt.figure(figsize=(8,8))
        images = []
        sess.run(init_op)
        X,y = sess.run(next_element)
        pp=p_joint(X,PERPLEXITY)
        sess.run(tf.initialize_all_variables())

        for i in range(1000):
            sess.run(init_op)
            X,y = sess.run(next_element)
            b,c,qq = sess.run([update_Y,C,Q],feed_dict={P:pp})
            xx,yy=b[-1].T
            print("iteration %d -- KL_Loss %f"%(i,c))

            if i%10==0:
                plt.title('KL_loss: {}'.format(c))
                img = [plt.scatter(b[-1][:,0],                            b[-1][:,1],                            c=y)]

                images.append(img)

        ani = ArtistAnimation(fig, images,interval=500)
        ani.save("class.mp4")
        plt.close('all')
