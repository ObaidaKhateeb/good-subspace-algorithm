# Good Subspace Algorithm
Implementation of the "Good Subspace" algorithm for the "Flat flatting problem", based on the paper "Efficient Subspace Approximation Algorithms" by N. Shyamalkumar & K. Varadarajan (2012).

## Problem Description 
Given n points in d-dimensional space, find a k-dimensional flat (affine subspace) that minimizes the sum of distances from points to the flat.

## Algorithm Description
The `Good-Subspace` algorithm is a recursive, randomized method aims to efficiently approximate the best-fitting k-dimensional subspace for a set of points. Its goal is to find a subspace such that the total distance from the points is at most 1+(ε^k) times the optimal. The process starts by constructing a set of candidate lines through the origin, each generated using probabilistic sampling where points are selected with probability proportional to their norm. One of these lines is selected uniformly at random and treated as the first direction in the subspace. The point set is then projected onto the orthogonal complement of this line, reducing the problem to a lower-dimensional space. The algorithm recurses on this projected data to find a (k-1)-dimensional subspace, and finally returns the span of the selected line and the recursively computed subspace.

Input: 

- Space dimension d
- Target dimension 1 ≤ k < d
- Error tolerance 0 < ε < 1
- Point multiset P ⊆ S (optional, if not provided, a random n-dimensional points are generated)

Output:

- A **formula** of the form:
    x(α) = [mean vector] + α₁*[basis vector 1] + α₂*[basis vector 2] + ...

    where the vectors define the k-dimensional subspace returned by the algorithm.
- **Approximation Quality Report** that includes optimal cost computed using PCA, the RD cost computed by our algorithm, and indication of whether the approximation guarantee was satisfied.
- **Visualization** (if the input is 2D or 3D) which includes the points and the fitted line/plane.

## Usage 
By CLI, using the following formats:

```bash
python good_subspace.py --k <target_dimension> --epsilon <epsilon> [--n <num_points> --d <dimension>] [--points <path_to_txt_file>]
```

## File Structure
    ├── good_subspace.py          # main implementation
    ├── report.pdf                # project representation, pdf format
    ├── report.docx               # project representation, word format
    ├── Efficient_Subspace.pdf    # the article the algorithm based on 
    ├── inputs/
        ├── points_d2.txt         # 2D points example of inpit
        ├── points_d3.txt         # 3D points example of inpit
        └── points_d3_2.txt       # 3D points example of inpit

## References 
Shyamalkumar, N. D., & Varadarajan, K. (2012). Efficient subspace approximation algorithms. Discrete & Computational Geometry, 47(1), 44-63.





 
