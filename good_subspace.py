import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #included in case matplotlib can't handle 3D 
import matplotlib.patches as mpatches
import argparse
import os

#A function that extracts points from a txt file 
#Input: a path for a txt file containing points (each point should be on a separate line)
#Output: a numpy array of the extracted points
def load_points(file_path):
    lines = os.path.join(file_path)
    with open(lines, 'r') as f:
        points = [list(map(float, line.strip().split())) for line in f if line.strip()]
    return np.array(points)

# A function that executes the Good Subspace algorithm
#Input: points P, dimension k, epsilon approximation factor
#Output: orthonormal basis of a k-dimensional subspace that approximates the best k-dimensional subspace for P
def good_subspace(P, k, epsilon):
    
    #Base case: random 1-direction (line through the origin)
    if k == 1:
        return pick_random_line(P, epsilon)
    
    #Generating O(1/epsilon * log(1/epsilon)) random directions
    c = 100 #constant chosen following a tuning process
    lines_count = int(np.ceil((c / epsilon) * np.log(1 / epsilon)))
    lines = []

    #l0 
    p = sample_point_norm_weighted(P)
    ell_0 = unit_vector(p)
    lines.append(ell_0)

    #Constructing the l0 to l_{lines_count} lines
    for j in range(1, lines_count):
        #Sampling a point from P
        r = sample_point_norm_weighted(P)
        v = unit_vector(r) #unit vector 
        u = lines[-1]  #unit vector of the last line

        if np.random.rand() < 0.5:
            candidate = (u,v)
        else:
            candidate =  (-u,v)

        t = np.random.rand()
        point = t * candidate[0] + (1 - t) * candidate[1]
        ell_j = unit_vector(point)
        lines.append(ell_j)

    #Picking a random line from the generated lines
    chosen_line = lines[np.random.randint(len(lines))]


    #Projecting points onto orthogonal complement of the line
    P_proj = project_onto_complement(P, chosen_line)

    #Recursively finding a good subspace in the projected space
    subspace_basis = good_subspace(P_proj, k - 1, epsilon)

    #Lifting the basis back to original space and add the chosen line
    full_basis = np.vstack([chosen_line, subspace_basis])
    return orthonormalize(full_basis)

# A function that picks 1D line that approxiamtes points in P 
#Input: points P, epsilon approximation factor
#Output: a unit vector representing the line
def pick_random_line(P, epsilon):
    lines_count = int(np.ceil((1 / epsilon) * np.log(1 / epsilon)))
    lines = []  
    for _ in range(lines_count):
        line = sample_point_norm_weighted(P)
        line = unit_vector(line)
        lines.append(line)

    return lines[np.random.randint(len(lines))]  # Uniform choice

#A function that samples a point from P with probability proportional to its norm
#Input: points P
#Output: a point from P
def sample_point_norm_weighted(P):
    norms = np.linalg.norm(P, axis=1)
    total = np.sum(norms)
    if total == 0:
        raise ValueError("All points have zero norm. Cannot perform norm-weighted sampling.")
    probs = norms / total
    index = np.random.choice(len(P), p=probs)
    return P[index]

# A function that computes the unit vector of a given vector
#Input: vector v
#Output: unit vector of v
def unit_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize the zero vector.")
    return v / norm

# A function that projects points onto the line spanned by vector v
#Input: points P, vector v
#Output: matrix of projected points
def project_onto_complement(P, v):
    return P - np.outer(P @ v, v)

#A function that orthonormalizes a set of vectors using the Gram-Schmidt process
#Input: list of vectors 
#Output: orthonormal basis of the subspace spanned by the vectors
def orthonormalize(V):
    Q = []
    for v in V:
        for q in Q:
            v -= np.dot(v, q) * q
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            Q.append(v / norm)
        else:
            return orthonormalize(V) #retry, for extreme cases where vectors are nearly linearly dependent
    return np.array(Q)

# A function that prints the parametric equation of the fitted subspace
#Input: mean vector mu, orthonormal basis of the subspace
#Output: prints the parametric equation of the fitted subspace
def print_subspace_formula(mu, basis):
    k, _ = basis.shape
    alpha_syms = [f"α{i+1}" for i in range(k)]

    #Build the mean vector 
    mu_str = "[" + ", ".join(f"{v:.4f}" for v in mu) + "]"

    #Build the direction terms
    direction_terms = []
    for i, vec in enumerate(basis):
        vec_str = "[" + ", ".join(f"{v:.4f}" for v in vec) + "]"
        direction_terms.append(f"{alpha_syms[i]}*{vec_str}")

    #Combine into the final formula
    formula = f"x(α) = {mu_str} + " + " + ".join(direction_terms)
    print("\nThe Fitted Subspace:")
    print(" ", formula)

# A function that computes the RD cost (Euclidean distance) between Points and subspace
#Input: points P, mean vector mu, orthonormal basis of the subspace
#Output: the RD cost
def compute_rd_cost(P, mu, basis, tau=2):
    diffs = P - mu
    projections = diffs @ basis.T @ basis
    residuals = diffs - projections
    dists = np.linalg.norm(residuals, axis=1)
    if tau == np.inf:
        return np.max(dists)
    return np.sum(dists ** tau) ** (1 / tau)

# A function that computes the optimal fitting cost using PCA
#Input: points P, dimension k
#Output: the optimal fitting cost
def optimal_pca_cost(P, k):
    centered = P - np.mean(P, axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    discarded = s[k:]  # discard top-k singular values
    return np.sqrt(np.sum(discarded**2))

# A function that visualizes the 3D points and the fitted 2D flat
def visualize_plane_3d(P, basis):
    mean_pt = np.mean(P, axis=0)
    grid_x, grid_y = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
    flat_pts = mean_pt + np.outer(grid_x.ravel(), basis[0]) + np.outer(grid_y.ravel(), basis[1])
    flat_pts = flat_pts.reshape(10, 10, 3)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='blue')
    ax.plot_surface(flat_pts[:, :, 0], flat_pts[:, :, 1], flat_pts[:, :, 2], alpha=0.5, color='orange')

    ax.set_title('GoodSubspace: Approximate 2D Flat in 3D')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(handles=[
        mpatches.Patch(color='blue', label='Data Points'),
        mpatches.Patch(color='orange', label='Fitted Subspace')
    ])
    plt.tight_layout()
    plt.show()

#A function that visualizes the 2D points and the fitted 1D line
def visualize_line_2d(P, basis):
    mean_pt = np.mean(P, axis=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(P[:, 0], P[:, 1], color='blue', label='Data Points')

    direction = basis[0]
    t = np.linspace(-3, 3, 100)
    line_pts = mean_pt + t[:, None] * direction
    ax.plot(line_pts[:, 0], line_pts[:, 1], color='orange', label='Fitted Line')

    ax.set_title('GoodSubspace: Approximate Line in 2D')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.show()

#A function that visualizes the 3D points and the fitted 1D line
def visualize_line_3d(P, basis):
    mean_pt = np.mean(P, axis=0)
    direction = basis[0]
    t = np.linspace(-3, 3, 100)
    line_pts = mean_pt + t[:, None] * direction

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='blue', label='Data Points')
    ax.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2], color='orange', label='Fitted Line')

    ax.set_title('GoodSubspace: Approximate Line in 3D')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.show()

def main(k, epsilon, n=None, d=None, points_path=None):
    #np.random.seed(42)

    #Extracts or generate points 
    if points_path:
        P = load_points(points_path)
        n, d = P.shape
    else:
        if n is None or d is None:
            raise ValueError("When points are not provided, both n and d are required.")
        base_plane = orthonormalize(np.random.randn(k, d))
        coeffs = np.random.randn(n, k)
        clean_points = coeffs @ base_plane
        noise = 0.05 * np.random.randn(n, d)
        P = clean_points + noise

    #Ensuring valid parameters
    if k >= d:
        raise ValueError("k must be less than d.")
    elif n < 0 or d < 0 or k < 0:
        raise ValueError("n, d, and k must be positive integers.")
    elif epsilon <= 0 or epsilon >= 1:
        raise ValueError("Epsilon must be in the range (0, 1).")

    #Fitting k-dimensional subspace 
    basis = good_subspace(P, k, epsilon)
    if basis.ndim == 1:
        basis = basis[np.newaxis, :]

    #Printing the fitted subspace formula
    mean_pt = np.mean(P, axis=0)
    print_subspace_formula(mean_pt, basis)

    #Computing the approximation cost and comparing with optimal PCA cost
    approx_cost = compute_rd_cost(P, mean_pt, basis)
    optimal_cost = optimal_pca_cost(P, k)
    bound = ((1 + epsilon) ** k) * optimal_cost
    print(f"\nOptimal Cost (PCA): {optimal_cost:.6f}")
    print(f"Allowed Bound: {(1 + epsilon) ** k:.4f} × {optimal_cost:.6f} = {bound:.6f}")
    print(f"RD Cost: {approx_cost:.6f}")
    if approx_cost <= bound:
        print("     ✅ The Algorithm satisfies the guarantee.")
    else:
        print(f"     ❌ The Algorithm does not satisfy the guarantee by {approx_cost - bound:.6f}.")

    #Visualizing the results (If applicable)
    if P.shape[1] == 3 and k == 2:
        visualize_plane_3d(P, basis)
    elif P.shape[1] == 3 and k == 1:
        visualize_line_3d(P, basis)
    elif P.shape[1] == 2 and k == 1:
        visualize_line_2d(P, basis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--d", type=int)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--points", type=str)

    args = parser.parse_args()
    main(args.k, args.epsilon, n=args.n, d=args.d, points_path=args.points)