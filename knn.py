import numpy as np
import matplotlib.pyplot as plt

##############################   SETUP   ###############################
### Magic numbers to play around with
K = 4   # K-closest
m = 20  # rows/users
n = 25  # cols/movies
d = 1   # Hidden factors, d << min(m, n)
rng_seed = 0

rng = np.random.default_rng(seed=rng_seed)
u = rng.random((m, d))
v = rng.random((n, d))

### Example:
"""
m = 3   # rows/users
n = 4   # cols/movies
K = 2   # K-closest
d = 2   # Hidden factors

                # action    # comedy
u = np.array([[ 0.9,        0.1],   # User 1
            [ 0.2,        0.8],   # User 2
            [ 0.6,        0.4]])  # User 3
v = np.array([[ 1.0,        0.1],   # action blockbuster
            [ 0.3,        0.9],   # romantic comedy
            [ 0.0,        0.9],   # pure romance
            [ 0.8,        0.6]])  # action/comedy
"""

# Hölder continuous function: $f(u_{ij}, v_{ij})$
# This is the latent structure for the signal matrix A
f = lambda u, v: np.tanh(np.dot(u, v.T)) + 2.5

# Signal matrix A
A = f(u, v)

# Noise matrix E
E = rng.random((m, n))

# Matrix of potential outcomes Y
Y = np.around(A + E, 2)

# TODO: I could also add the propensity matrix P but numpy already is
#       taking care of that for me. Besides, it's not needed for KNN.

# Missingness mask matrix D
D = rng.integers(low = 0, high = 2, size = (m, n))

# Partially-obscured outcome matrix $\tilde{Y}$
Yt = np.copy(Y)
Yt[D==0] = np.nan
#Yt = np.ma.masked_array(Y, D==0, fill_value=np.nan)    # D==0 for now

print("True values:")
print(Y)
print("Masked:")
print(Yt)

def KNN(Yt: np.ndarray, K: int):
    """
    Takes an array and performs KNN algorithm to find missing values
    denoted by np.nan. How many neighbors (K) is configurable. You
    don't need to worry about choosing a K that's too high, the
    function deals with that for you. Returns a copy of the same array but with
    the missing values imputed, if possible.
    """

    # Disobscured outcome matrix to be returned
    Z = np.copy(Yt)

    ###############################   KNN   ################################
    # Distance function to define "closeness" (mean-squared distance)
    def distance(A: np.ndarray, i: int, r: int):
        # use only common values between rows
        common = ~np.isnan(A[i]) & ~np.isnan(A[r])
        # set nan if there aren't any true (common) values
        if not np.any(common):
            return np.nan
        return np.mean( (A[i, common]-A[r, common]) ** 2)

    # For each row that is missing a value
    closeness_dict = {}
    nrow = np.shape(Yt)[0]
    for row in range(0, nrow):
        if np.any(np.isnan(Yt[row])):
            # Add other rows' distance to the dictionary
            distances = [(i, distance(Yt, row, i)) for i in range(0, nrow) if i != row]
            closeness_dict[row] = distances

    # Sort the closeness_dict
    for _ in closeness_dict.values():
        _.sort(key=lambda x: x[1])

    # Select the K closest (that aren't missing the same entry)
    # and compute their average
    for i, j in np.argwhere(np.isnan(Yt)):
        avg = []
        k = 0
        rows_left = nrow - 1    # b/c excluding 1 row (itself)
        for row in range(0, rows_left):
            # break if K nearest neighbors have been reached
            if k == K:
                break
            # check to see if the k-st closest row to row i isn't also 
            # missing an entry
            candidate_row = closeness_dict[i][row][0]
            if np.isnan(Yt[candidate_row, j]) == False:
                avg.append(Yt[candidate_row, j])
                k += 1

        Z[i, j] = np.mean(avg)

    # Small note to let us know if K is too big
    if rows_left == 0:
        K = nrow - 1
        raise Warning("K is larger than the number of rows")

    print(f"Imputed with {K=}")
    print(Z)

    return Z

KNN(Yt, K)

badness = float(np.mean(np.abs(Y-KNN(Yt, K))))
print(f"The badness is {badness}")

##################################   PLOTS   ##################################
fig, ax = plt.subplots(2, 2)

plot0 = ax[0, 0].imshow(Y, cmap='gray_r', vmin=np.min(Y)-np.mean(Y)/10, aspect='equal')
ax[0, 0].set_title("Array before being hidden")
ax[0, 0].axis('off')
plt.colorbar(plot0, ax=ax[0, 0])

plot1 = ax[0, 1].imshow(Yt, cmap='gray_r', vmin=np.min(Y)-np.mean(Y)/10, aspect='equal')
ax[0, 1].set_title("Array after being hidden")
ax[0, 1].axis('off')
plt.colorbar(plot1, ax=ax[0, 1])

plot2 = ax[1, 0].imshow(KNN(Yt, K), cmap='gray_r', vmin=np.min(Y)-np.mean(Y)/10, aspect='equal')
ax[1, 0].set_title("Imputed array")
ax[1, 0].axis('off')
plt.colorbar(plot2, ax=ax[1, 0])

plot3 = ax[1, 1].imshow(np.abs(Y-KNN(Yt, K)), cmap='gray_r', aspect='equal')
ax[1, 1].set_title("Difference (absolute value)")
ax[1, 1].axis('off')
plt.colorbar(plot3, ax=ax[1, 1])

plt.show()