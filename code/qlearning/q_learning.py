from environment import MountainCar
import sys
import numpy as np


############################## Helper Math #####################################
def sigmoid(z):
    return (1.0 / (1 + np.exp(-z)))

def sparse_dot(X, Y):
    """
    Takes a 'sparse array' X (in dict form as below) and computes the dot
    product with the normal list-like array Y.

    Parameters
    ----------
    X : dict
        Of the form {idx : value} where idx is the index in an equivalent
        dense array. Only nonzero elements are contained here.
    Y : list-like
        The vector to dot with X, contains real values.

    Returns
    -------
    float
        The scalar dot product of the two vectors.
    """
    product = 0.0
    for i, x in X.items():
        product += x * Y[i]
    return product

def sparse_add(X, Y):
    """Expecting X to be the sparse one."""
    Y_new = Y.copy()
    for i, x in X.items():
        Y_new[i] += x
    return Y_new

def sparse_sub(X, Y):
    """Expecting X to be the sparse one. Returns Y-X."""
    Y_new = Y.copy()
    for i, x in X.items():
        Y_new[i] -= x
    return Y_new

def sparse_mult(X, Y):
    """Expecting X to be the sparse one."""
    Y_new = Y.copy()
    for i, x in X.items():
        Y_new[i] *= x
    return Y_new

def sparse_scalar_mult(s, X):
    """X is the sparse vector, s is the scalar."""
    return {i: v*s for i, v in X.items()}


def q(s, w_a, bias):
    """Expecting s to be a sparse state vector, a to be full np array.
        Bias is a scalar of course."""
    return sparse_dot(s, w_a) + bias

def optimal(s, w):
    q_array = np.zeros(3)
    
    for i in range(3):
        q_array[i] = sparse_dot(s, w[:,i])

    return np.argmax(q_array)
    
def compute_error(X, Y):
    return np.mean(np.array(X) != np.array(Y))



############################## File I/O #######################################
def write_weights(filename, b, w):
    with open(filename, 'w') as file:
        file.write(str(b) + '\n')
        for x in w.flatten():
            file.write(str(x) + '\n')
            
def write_returns(filename, rewards):
    with open(filename, 'w') as file:
        for r in rewards:
            file.write(str(r) + '\n')




############################## Main method ####################################
def main(args):
    if len(args) != 9:
        print("Wrong number of arguments!\n"
          + "Correct usage is 'python q_learning.py <args... (8 args)>'")
        sys.exit(1)
        
    mode = args[1]
    weight_out = args[2]
    returns_out = args[3]
    episodes = int(args[4])
    max_iterations = int(args[5])
    epsilon = float(args[6])
    gamma = float(args[7])
    alpha = float(args[8])
    
    # initialize the environment plus a RNG
    env = MountainCar(mode=mode)
    rand = np.random
    
    # initialize weights and bias
    w = np.zeros((env.state_space, env.action_space))
    b = 0.0
    
    # init rewards
    rewards = []
    
    # now let's learn
    for e in range(episodes):           # for each episode
        done = False                        # set flag to false        
        s = env.reset()                     # get the initial state
        reward = 0.0                        # initialize reward for this episode
        
        # Start your engines
        for i in range(max_iterations):     # for each iteration
            if done:                            # if we reached the goal
                break                               # break and go to next episode
            
            # Get next action
            if rand.random() < epsilon:           # epsilon-greedy check
                a = rand.choice(3)                  # random choice
            else:
                a = optimal(s, w)                   # otherwise get optimal choice
                
            # Now take that action
            s_prime, r, done = env.step(a)          # call env for new state, reward, and flag
            
            # Update reward
            reward += r
            
            # Update the weights (q-values) and bias
            # 1. Calculate the innermost quantity (difference) using old weights and bias
            diff = q(s, w[:,a], b) - (r + gamma*(q(s_prime, w[:,optimal(s_prime, w)], b)))
            
            # 2. Now multiply by learning rate -- this is the bias update term
            coeff = alpha*diff
            
            # 3. Multiply by the gradient -- this is the weight update vector
            update = sparse_scalar_mult(coeff, s)
            
            # 4. Finally actually update both
            b -= coeff
            w[:,a] = sparse_sub(update, w[:,a])
            
            # Now update s!
            s = s_prime.copy()
        
        # Once we finish the episode
        rewards.append(reward)                  # append this reward
        
    # After all episodes
    write_returns(returns_out, rewards)         # output rewards
    write_weights(weight_out, b, w)             # output bias and weights
    
    
    ###  EMPIRICAL QUESTIONS
    x = np.arange(episodes)
    y1 = np.array(rewards)
    y2 = np.zeros(len(rewards))
    for idx in range(25, len(rewards)):
        y2[idx] = np.sum(y1[idx-25:idx]) / 25
        
    # now plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    
    # plot both sets
    plt.plot(x, y1, linestyle='-', alpha=0.5, color='C0', label = 'Episode return')
    plt.plot(x, y2, linestyle='-', alpha=0.5, color='C1', label = '25-episode average return')
    
    plt.legend(title = 'Raw vs. Avg', loc = 'best')
    plt.title("Q-learning return per episode with tile state representation")
    plt.xlabel('Episode')
    plt.ylabel('Return value')
    plt.savefig('plot.png')
    plt.show()
    
    
    ### Debugging output
    print(rewards)
    print(b)        # output bias
    print(w)        # output weights
    

if __name__ == "__main__":
    main(sys.argv)
    
    


    
    
    