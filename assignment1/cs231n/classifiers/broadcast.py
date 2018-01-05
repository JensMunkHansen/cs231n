import numpy as np

a = np.array([[1,1,1,1],[2,2,2,2]])
b = np.array([[1,2,3,4],[1,1,1,1],[1,2,1,9]])
np.linalg.norm(a[:, np.newaxis] - b, axis = 2)



Here are the original input variables:

A = np.array([[1,1,1,1],[2,2,2,2]])
B = np.array([[1,2,3,4],[1,1,1,1],[1,2,1,9]])
A
# array([[1, 1, 1, 1],
#        [2, 2, 2, 2]])
B
# array([[1, 2, 3, 4],
#        [1, 1, 1, 1],
#        [1, 2, 1, 9]])
A is a 2x4 array. B is a 3x4 array.

We want to compute the Euclidean distance matrix operation in one entirely vectorized operation, where dist[i,j] contains the distance between the ith instance in A and jth instance in B. So dist is 2x3 in this example.

The distance

dist(A,B) = sqrt((A-B)**2)

could ostensibly be written with numpy as

dist = np.sqrt(np.sum(np.square(A-B))) # DOES NOT WORK
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# ValueError: operands could not be broadcast together with shapes (2,4) (3,4)
However, as shown above, the problem is that the element-wise subtraction operation A-B involves incompatible array sizes, specifically the 2 and 3 in the first dimension.

A has dimensions 2 x 4
B has dimensions 3 x 4
In order to do element-wise subtraction, we have to pad either A or B to satisfy numpy's broadcast rules. I'll choose to pad A with an extra dimension so that it becomes 2 x 1 x 4, which allows the arrays' dimensions to line up for broadcasting. For more on numpy broadcasting, see the tutorial in the scipy manual and the final example in this tutorial.

You can perform the padding with either np.newaxis value or with the np.reshape command. I show both below:

# First approach is to add the extra dimension to A with np.newaxis
A[:,np.newaxis,:] has dimensions 2 x 1 x 4
B has dimensions                     3 x 4

# Second approach is to reshape A with np.reshape
np.reshape(A, (2,1,4)) has dimensions 2 x 1 x 4
B has dimensions                          3 x 4
As you can see, using either approach will allow the dimensions to line up. I'll use the first approach with np.newaxis. So now, this will work to create A-B, which is a 2x3x4 array:

diff = A[:,np.newaxis,:] - B
# Alternative approach:
# diff = np.reshape(A, (2,1,4)) - B
diff.shape
# (2, 3, 4)
Now we can put that difference expression into the dist equation statement to get the final result:

dist = np.sqrt(np.sum(np.square(A[:,np.newaxis,:] - B), axis=2))
dist
# array([[ 3.74165739,  0.        ,  8.06225775],
#        [ 2.44948974,  2.        ,  7.14142843]])
Note that the sum is over axis=2, which means take the sum over the 2x3x4 array's third axis (where the axis id starts with 0).

If your arrays are small, then the above command will work just fine. However, if you have large arrays, then you may run into memory issues. Note that in the above example, numpy internally created a 2x3x4 array to perform the broadcasting. If we generalize A to have dimensions a x z and B to have dimensions b x z, then numpy will internally create an a x b x z array for broadcasting.

We can avoid creating this intermediate array by doing some mathematical manipulation. Because you are computing the Euclidean distance as a sum-of-squared-differences, we can take advantage of the mathematical fact that sum-of-squared-differences can be rewritten.

dist(A,B) = sqrt((A-B)**2)
          = sqrt(A**2 - 2*A*B + B**2)

Note that the middle term involves the sum over element-wise multiplication. This sum over multiplcations is better known as a dot product. Because A and B are each a matrix, then this operation is actually a matrix multiplication. We can thus rewrite the above as:

dist(A,B) = sqrt(A**2 - 2*A*B + B**2)

We can then write the following numpy code:

threeSums = np.sum(np.square(A)[:,np.newaxis,:], axis=2) - 2 * A.dot(B.T) + np.sum(np.square(B), axis=1)
dist = np.sqrt(threeSums)
dist
# array([[ 3.74165739,  0.        ,  8.06225775],
#        [ 2.44948974,  2.        ,  7.14142843]])
Note that the answer above is exactly the same as the previous implementation. Again, the advantage here is the we do not need to create the intermediate 2x3x4 array for broadcasting.

For completeness, let's double-check that the dimensions of each summand in threeSums allowed broadcasting.

np.sum(np.square(A)[:,np.newaxis,:], axis=2) has dimensions 2 x 1
2 * A.dot(B.T) has dimensions                               2 x 3
np.sum(np.square(B), axis=1) has dimensions                 1 x 3
So, as expected, the final dist array has dimensions 2x3.

This use of the dot product in lieu of sum of element-wise multiplication is also discussed in this tutorial.
