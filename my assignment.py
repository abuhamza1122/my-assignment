#!/usr/bin/env python
# coding: utf-8

# ### 1) What is a vector?
# #### a quantity having direction as well as magnitude, especially as determining the position of one point in space relative to another.

# ### 2) How do you represent vectors using a Python list? Give an example.
# 

# In[3]:


# You can represent vectors in Python using lists, where each element of the list corresponds to a component of the vector.
#Here's an example of how to represent a 2D vector using a Python list:
# Representing a 2D vector [x, y]
vector_2d = [3, 4]  # This represents a 2D vector with x = 3 and y = 4


# ### 3) What is a dot product of two vectors?
# #### The dot product, also called scalar product, is a measure of how closely two vectors align, in terms of the directions they point.

# ### 4) Write a function to compute the dot product of two vectors.
# 

# In[4]:


def dot_product(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same dimension for dot product calculation.")
    
    result = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
    return result

# Example usage:
vector1 = [3, 4, 5]
vector2 = [1, 2, 3]

result = dot_product(vector1, vector2)
print("Dot Product:", result)


# ### 5) what is numpy?
# #### NumPy is a Python library used for working with arrays.

# ### 6) How do you install Numpy?
# #### You have to write this in the command line "pip install numpy"
# 

# ### 7) How do you import the numpy module?
# #### import numpy

# ### 8) What does it mean to import a module with an alias? Give an example.
# #### Import aliases are where you take your standard import, but instead of using a pre-defined name by the exporting module, you use a name that is defined in the importing module.

# In[5]:


import numpy as np


# ### 9) What is the commonly used alias for numpy?
# #### NumPy is usually imported under the np alias. 

# ### 10) What is a Numpy array?
# #### A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. 

# ### 11) How do you create a Numpy array? Give an example.
# 

# In[5]:


# Once you have NumPy installed, you can create NumPy arrays using the numpy.array() function. Here's an example of how 
# to create a simple NumPy array:
import numpy as np

# Creating a NumPy array from a Python list
my_list = [1, 3, 5, 7, 9]
my_array = np.array(my_list)

print(my_array)


# ### 12) What is the type of Numpy arrays?
# #### booleans (bool), integers (int), unsigned integers (uint) floating point (float) and complex.

# ### 13) How do you access the elements of a Numpy array?

# In[6]:


# First, you need to import the numpy library, and then you can create an array using the numpy.array() function.
# Here's an example:
import numpy as np

# Create a NumPy array from a Python list
my_list = [12, 24, 36, 48, 60]
my_array = np.array(my_list)

print(my_array)


# In[7]:


import numpy as np

# Create a 2D NumPy array from a list of lists
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(matrix)


# In[9]:


import numpy as np

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

print(arr[0, 1, 2])


# ### 14) How do you compute the dot product of two vectors using Numpy?
# 

# In[9]:


#  We can compute the dot product of two vectors using NumPy's numpy.dot() function or the @ operator for element-wise
# multiplication. Here is only one example:

#  Using numpy.dot()
import numpy as np

# Define two NumPy arrays representing vectors
vector1 = np.array([10, 12, 43])
vector2 = np.array([41, 25, 36])

# Compute the dot product using numpy.dot()
dot_product = np.dot(vector1, vector2)

print("Dot Product (Method 1):", dot_product)


# ### 15) What happens if you try to compute the dot product of two vectors which have different sizes?
# #### The dot product is applicable only for the pairs of vectors that have the same number of dimensions.

# ### 16) How do you compute the element-wise product of two Numpy arrays?

# In[10]:


import numpy as np

# Define two NumPy arrays
array1 = np.array([11, 22, 53])
array2 = np.array([49, 55, 64])

# Compute the element-wise product using the * operator
elementwise_product = array1 * array2

print("Element-Wise Product:", elementwise_product)


# ### 17) How do you compute the sum of all the elements in a Numpy array?
# 

# In[12]:


import numpy as np

# Define a NumPy array
my_array = np.array([1, 2, 3, 4, 5])

# Compute the sum using the array.sum() method
array_sum = my_array.sum()

print("Sum :", array_sum)


# ### 18) What are the benefits of using Numpy arrays over Python lists for operating on numerical data?
# #### NumPy arrays are faster than Python lists.

# ### 19) Why do Numpy array operations have better performance compared to Python functions and loops?
# #### A Python list, however, is only a collection of objects. A NumPy array allows only homogeneous data types. 

# ### 20) Illustrate the performance difference between Numpy array operations and Python loops using an example.
# 

# In[13]:


import numpy as np
import time

# Create large arrays with 10 million elements
array_size = 10**7
numpy_array = np.arange(array_size)
python_list = list(range(array_size))

# Using NumPy for element-wise multiplication
start_time = time.time()
numpy_result = numpy_array * numpy_array
numpy_time = time.time() - start_time

# Using Python loop for element-wise multiplication
start_time = time.time()
python_result = [x * x for x in python_list]
python_time = time.time() - start_time

# Check if the results match
results_match = np.array_equal(numpy_result, python_result)

print(f"NumPy Execution Time: {numpy_time:.6f} seconds")
print(f"Python Loop Execution Time: {python_time:.6f} seconds")
print("Results Match:", results_match)


# ### 21) What are multi-dimensional Numpy arrays?
# #### A multi-dimensional array is an array with more than one level or dimension.

# ### 22) Illustrate the creation of Numpy arrays with 2, 3, and 4 dimensions.

# In[15]:


# 2-D dimensions
import numpy as np

arr = np.array([[2, 4, 6], [8, 10, 12]])

print(arr)
print('number of dimensions :', arr.ndim)


# In[16]:


# 3-D dimensions
import numpy as np

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

print(arr)
print('number of dimensions :', arr.ndim)


# In[17]:


# 4-D dimensions
import numpy as np

arr = np.array([10, 20, 30, 44], ndmin=4)

print(arr)
print('number of dimensions :', arr.ndim)


# ### 23) How do you inspect the number of dimensions and the length along each dimension in a Numpy array?
# 

# In[18]:


import numpy as np

# Create a NumPy array
my_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Inspect the number of dimensions
num_dimensions = my_array.ndim

# Inspect the length along each dimension
dimensions = my_array.shape

print("Number of Dimensions:", num_dimensions)
print("Length along Each Dimension:", dimensions)


# ### 24) Can the elements of a Numpy array have different data types?
# #### The elements of a NumPy array must all be of the same type

# ### 25) How do you check the data type of the elements of a Numpy array?

# In[19]:


import numpy as np

arr = np.array([13, 26, 39, 52], ndmin=5)

print(arr.dtype)


# ### 26) What is the data type of a Numpy array?
# #### Here's the list of most commonly used numeric data types in NumPy: int8 , int16 , int32 , int64 - signed integer types with different bit sizes.

# ### 27) What is the difference between a matrix and a 2D Numpy array?
# #### Numpy arrays (nd-arrays) are N-dimensional where, N=1,2,3… Numpy matrices are strictly 2-dimensional. 

# ### 28) How do you perform matrix multiplication using Numpy?

# In[20]:


import numpy as np

X = 5

Y = [[1, 7],
      [6, 10]]
 
print(np.dot(X,Y))


# ### 29) What is the @ operator used for in Numpy?
# #### The @ operator, available since Python 3.5, can be used for conventional matrix multiplication.

# ### 30) What is the CSV file format?
# #### A CSV file is a spreadsheet format, so it can be opened by spreadsheet applications like Microsoft Excel and Google Spreadsheets. Since CSV files are used to exchange large volumes of data, database programs, analytical software, and applications that can store massive amounts of information usually support the CSV.

# ### 31) How do you read data from a CSV file using Numpy?

# In[23]:


import numpy as np
 
# using loadtxt()
arr = np.loadtxt("sample_data.csv",
                 delimiter=",", dtype=str)
display(arr)


# In[24]:


import numpy as np
 
# using genfromtxt()
arr = np.genfromtxt("sample_data.csv",
                    delimiter=",", dtype=str)
display(arr)


# ### 32) How do you concatenate two Numpy arrays?

# In[25]:


import numpy as np

# Create two NumPy arrays
array1 = np.array([2, 4, 6])
array2 = np.array([5, 3, 1])

# Concatenate the arrays along axis 0 (default, results in a new 1D array)
concatenated_array = np.concatenate((array1, array2))

print("Concatenated Array:")
print(concatenated_array)


# ### 33) What is the purpose of the axis argument of np.concatenate?
# #### Numpy concatenate is like “stacking” numpy arrays
# #### The axis that we specify with the axis parameter is the axis along which we stack the arrays. 

# ### 34) When are two Numpy arrays compatible for concatenation?
# #### The arrays must have the same shape, except in the dimension corresponding to axis.

# ### 35) Give an example of two Numpy arrays that can be concatenated.
# 

# In[26]:


import numpy as np

# Create two NumPy arrays with the same number of columns (2D arrays)
array1 = np.array([[1, 2], [3, 4]])
array2 = np.array([[5, 6], [7, 8]])

# Concatenate the arrays along rows (axis=0)
concatenated_array = np.concatenate((array1, array2), axis=0)

print("Concatenated 2D Array (along rows, axis=0):")
print(concatenated_array)



# ### 36) Give an example of two Numpy arrays that cannot be concatenated.
# 

# In[28]:


x = np.array([11, 22])
y = np.array([36, 44])
np.concatenate(x, y)

# TypeError: only length-1 arrays can be converted to Python scalars


# ### 37) What is the purpose of the np.reshape function?
# #### Change an Array's Shape Using NumPy reshape() NumPy's reshape() enables you to change the shape of an array into another compatible shape. 

# ### 38) What does it mean to “reshape” a Numpy array?
# #### Reshaping means changing the shape of an array. The shape of an array is the number of elements in each dimension. By reshaping we can add or remove dimensions or change number of elements in each dimension.

# ### 39) How do you write a numpy array into a CSV file?
# 

# In[32]:


# save numpy array as csv file
from numpy import asarray
from numpy import savetxt
# define data
data = asarray([[ 10, 20, 30, 40, 50, 60, 70, 80, 90]])
# save to csv file
savetxt('data.csv', data, delimiter=',')


# ### 40) Give some examples of Numpy functions for performing mathematical operations.

# In[34]:


import numpy as np

# Example 1: Basic Arithmetic Operations
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
addition = np.add(arr1, arr2)
subtraction = np.subtract(arr1, arr2)
multiplication = np.multiply(arr1, arr2)
division = np.divide(arr1, arr2)

# Example 2: Trigonometric Functions
angles = np.array([0, np.pi/4, np.pi/2])
sin_values = np.sin(angles)
cos_values = np.cos(angles)

# Example 3: Logarithmic and Exponential Functions
arr3 = np.array([1, 10, 100])
log_values = np.log(arr3)
exp_values = np.exp(arr3)

# Example 4: Linear Algebra
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
dot_product = np.dot(matrix1, matrix2)
matrix_multiply = np.matmul(matrix1, matrix2)

print("Example 1 - Addition:", addition)
print("Example 1 - Subtraction:", subtraction)
print("Example 1 - Multiplication:", multiplication)
print("Example 1 - Division:", division)

print("Example 2 - Sine Values:", sin_values)
print("Example 2 - Cosine Values:", cos_values)

print("Example 3 - Natural Logarithm:", log_values)
print("Example 3 - Exponential Values:", exp_values)

print("Example 4 - Dot Product:", dot_product)
print("Example 4 - Matrix Multiplication:\n", matrix_multiply)


# ### 41) Give some examples of Numpy functions for performing array manipulation.
# 

# In[35]:


import numpy as np

# Create two arrays for demonstration
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])

# Reshaping Arrays
reshaped_arr = np.reshape(arr1, (3, 2))
transposed_arr = np.transpose(arr1)

# Concatenation and Stacking
concatenated_arr = np.concatenate((arr1, arr2), axis=0)
stacked_vertically = np.vstack((arr1, arr2))
stacked_horizontally = np.hstack((arr1, arr2))

# Splitting Arrays
split_arr = np.split(reshaped_arr, 3)
vsplit_arr = np.vsplit(concatenated_arr, 2)
hsplit_arr = np.hsplit(concatenated_arr, 3)

# Sorting
sorted_arr = np.sort(arr1, axis=None)

print("Original Array 1:\n", arr1)
print("Reshaped Array:\n", reshaped_arr)
print("Transposed Array 1:\n", transposed_arr)
print("Concatenated Array:\n", concatenated_arr)
print("Vertically Stacked Array:\n", stacked_vertically)
print("Horizontally Stacked Array:\n", stacked_horizontally)
print("Split Arrays:\n", split_arr)
print("Vertically Split Arrays:\n", vsplit_arr)
print("Horizontally Split Arrays:\n", hsplit_arr)
print("Sorted Array 1:\n", sorted_arr)


# ### 42) Give some examples of Numpy functions for performing linear algebra.
# 

# In[36]:


import numpy as np

# Example matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
x = np.array([1, 2])

# Matrix multiplication
result_dot = np.dot(A, B)
result_matmul = np.matmul(A, B)
result_at_operator = A @ B

# Matrix inverse and determinant
inv_A = np.linalg.inv(A)
det_A = np.linalg.det(A)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Singular Value Decomposition (SVD)
U, S, VT = np.linalg.svd(A)

# Linear equation solving
coeff_matrix = np.array([[2, 3], [1, -2]])
constants = np.array([8, 1])
solution = np.linalg.solve(coeff_matrix, constants)

# Matrix norms
Frobenius_norm = np.linalg.norm(A, 'fro')
L2_norm_vector = np.linalg.norm(x, ord=2)

print("Matrix Multiplication (Dot):", result_dot)
print("Matrix Multiplication (Matmul):", result_matmul)
print("Matrix Multiplication (@ Operator):", result_at_operator)
print("Matrix Inverse:", inv_A)
print("Determinant of A:", det_A)
print("Eigenvalues of A:", eigenvalues)
print("Eigenvectors of A:", eigenvectors)
print("Singular Value Decomposition - U:", U)
print("Singular Value Decomposition - S:", S)
print("Singular Value Decomposition - VT:", VT)
print("Linear Equation Solution:", solution)
print("Frobenius Norm of A:", Frobenius_norm)
print("L2 Norm of x:", L2_norm_vector)


# ### 43) Give some examples of Numpy functions for performing statistical operations.
# 

# In[37]:


import numpy as np

# Example data
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Mean and Median
mean = np.mean(data)
median = np.median(data)

# Variance and Standard Deviation
variance = np.var(data)
std_deviation = np.std(data)

# Min and Max
min_value = np.min(data)
max_value = np.max(data)

# Sum and Product
sum_values = np.sum(data)
product_values = np.prod(data)

print("Mean:", mean)
print("Median:", median)
print("Variance:", variance)
print("Standard Deviation:", std_deviation)
print("Min Value:", min_value)
print("Max Value:", max_value)
print("Sum of Values:", sum_values)
print("Product of Values:", product_values)


# ### 44) How do you find the right Numpy function for a specific operation or use case?
# 

# In[38]:


import numpy as np

# Example array of data
data = np.array([10, 20, 30, 40, 50])

# Calculate the standard deviation
std_deviation = np.std(data)

print("Standard Deviation:", std_deviation)


# ### 45) Where can you see a list of all the Numpy array functions and operations?
# 

# In[46]:


import numpy as np
dir(np)


# ### 46) What are the arithmetic operators supported by Numpy arrays? Illustrate with examples.

# In[47]:


import numpy as np

first_array = np.array([1, 3, 5, 7])
second_array = np.array([2, 4, 6, 8])

# using the + operator
result1 = first_array + second_array
print("Using the + operator:",result1) 

# using the add() function
result2 = np.add(first_array, second_array)
print("Using the add() function:",result2)


# In[48]:


import numpy as np

first_array = np.array([3, 9, 27, 81])
second_array = np.array([2, 4, 8, 16])

# using the - operator
result1 = first_array - second_array
print("Using the - operator:",result1) 

# using the subtract() function
result2 = np.subtract(first_array, second_array)
print("Using the subtract() function:",result2) 


# In[49]:


import numpy as np

first_array = np.array([1, 3, 5, 7])
second_array = np.array([2, 4, 6, 8])

# using the * operator
result1 = first_array * second_array
print("Using the * operator:",result1) 

# using the multiply() function
result2 = np.multiply(first_array, second_array)
print("Using the multiply() function:",result2) 


# In[50]:


import numpy as np

first_array = np.array([1, 2, 3])
second_array = np.array([4, 5, 6])

# using the / operator
result1 = first_array / second_array
print("Using the / operator:",result1) 

# using the divide() function
result2 = np.divide(first_array, second_array)
print("Using the divide() function:",result2) 


# In[51]:


import numpy as np

array1 = np.array([1, 2, 3])

# using the ** operator
result1 = array1 ** 2
print("Using the ** operator:",result1) 

# using the power() function
result2 = np.power(array1, 2)
print("Using the power() function:",result2) 


# In[52]:


import numpy as np

first_array = np.array([9, 10, 20])
second_array = np.array([2, 5, 7])

# using the % operator
result1 = first_array % second_array
print("Using the % operator:",result1) 

# using the mod() function
result2 = np.mod(first_array, second_array)
print("Using the mod() function:",result2)


# ### 47) What is array broadcasting? How is it useful? Illustrate with an example.
# #### The term broadcasting describes how numpy treats arrays with different shapes during arithmetic operations.
# #### Broadcasting provides a means of vectorizing array operations so that looping occurs in C instead of Python. It does this without making needless copies of data and usually leads to efficient algorithm implementations.

# ### 48) Give some examples of arrays that are compatible for broadcasting? 

# In[40]:


import numpy as np

# Create a 1D array
arr = np.array([1, 2, 3, 4, 5])

# Add a scalar to the array
result = arr + 10

print(result)



# ### 49) Give some examples of arrays that are not compatible for broadcasting?
# 

# In[42]:


import numpy as np

arr1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 3x3 array
arr2 = np.array([[10, 20], [30, 40]])  # 2x2 array

result = arr1 + arr2  # Broadcasting not possible


# ### 50) What are the comparison operators supported by Numpy arrays? Illustrate with examples.
# 

# In[43]:


import numpy as np

# Create two NumPy arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([3, 2, 1, 4, 5])

# Element-wise comparisons
equal_to = arr1 == arr2
not_equal_to = arr1 != arr2
less_than = arr1 < arr2
less_than_or_equal_to = arr1 <= arr2
greater_than = arr1 > arr2
greater_than_or_equal_to = arr1 >= arr2

print("Equal To:", equal_to)
print("Not Equal To:", not_equal_to)
print("Less Than:", less_than)
print("Less Than or Equal To:", less_than_or_equal_to)
print("Greater Than:", greater_than)
print("Greater Than or Equal To:", greater_than_or_equal_to)


# ### 51) How do you access a specific subarray or slice from a Numpy array?
# 

# In[44]:


import numpy as np

# Create a NumPy array
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Slice from index 2 to 6 (exclusive)
subarray = arr[2:6]

print(subarray)


# ### 52) Illustrate array indexing and slicing in multi-dimensional Numpy arrays with some examples.
# 

# In[46]:


import numpy as np

# Create a 2D NumPy array
matrix = np.array([[10, 20, 30],
                  [40, 50, 60],
                  [70, 80, 90]])

# Print an element at a specific position
element = matrix[1, 2]  # Row 1 (second row), Column 2 (third column)

print("Element at (1, 2):", element)


# ### 53) How do you create a Numpy array with a given shape containing all zeros?
# 

# In[48]:


np.zeros((5,8))


# ### 54) How do you create a Numpy array with a given shape containing all ones?
# 

# In[67]:


np.ones((2,4))


# ### 55) How do you create an identity matrix of a given shape?
# 

# In[49]:


import numpy as np

# Create a 3x3 identity matrix
identity_matrix = np.eye(3)

print(identity_matrix)


# ### 56) How do you create a random vector of a given length?

# In[50]:


import numpy as np

# Specify the length of the random vector
vector_length = 5

# Create a random vector of the specified length
random_vector = np.random.rand(vector_length)

print(random_vector)


# ### 57) How do you create a Numpy array with a given shape with a fixed value for each element?
# 

# In[52]:


import numpy as np

b1 = np.zeros(6)
print(b1)


# ### 58) How do you create a Numpy array with a given shape containing randomly initialized elements?
# 

# In[53]:


import numpy as np

# Specify the desired shape of the array
array_shape = (3, 4)  # Example: 3 rows and 4 columns

# Create a NumPy array with random values in the specified shape
random_array = np.random.rand(*array_shape)

print(random_array)


# ### 59) What is the difference between np.random.rand and np.random.randn? Illustrate with examples.
# ### numpy.random.randn generates samples from the normal distribution, while numpy.random.rand from uniform (in range [0,1)).

# In[55]:


import numpy as np
import matplotlib.pyplot as plt

sample_size = 200000
uniform = np.random.rand(sample_size)
normal = np.random.randn(sample_size)

pdf, bins, patches = plt.hist(uniform, bins=20, range=(0, 1), density=True)
plt.title('rand: uniform')
plt.show()

pdf, bins, patches = plt.hist(normal, bins=20, range=(-4, 4), density=True)
plt.title('randn: normal')
plt.show()


# ### 60) What is the difference between np.arange and np.linspace? Illustrate with examples.
# #### np.arange : Return evenly spaced values within a given interval.
# #### np.linspace : Return evenly spaced numbers over a specified interval.
# 

# In[56]:


np.linspace(0,2,9)
np.array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])


# In[57]:


np.arange(0,1,.9)
np.array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


# ### 1. Import the numpy package under the name np

# In[79]:


import numpy as np


# ### 2. Print the numpy version and the configuration

# In[80]:


import numpy as np
print(np.__version__)
print(np.show_config())


# ### 3. Create a null vector of size 10

# In[81]:


import numpy as np
x = np.zeros(10)
print(x)


# ### 4. How to find the memory size of any array

# In[59]:


import numpy as np

# Create a NumPy array
arr = np.array([1, 2, 3, 4, 5])

# Get the memory size of the array in bytes
memory_size = arr.itemsize * arr.size

print("Memory size of the array (in bytes):", memory_size)




# ### 5. How to get the documentation of the numpy add function from the command line?

# In[84]:


import numpy as np
print(np.info(np.add))


# ### 6. Create a null vector of size 10 but the fifth value which is 1

# In[60]:


x=np.zeros(10)
x[4]=1
print(x)


# ### 7. Create a vector with values ranging from 10 to 49

# In[61]:


import numpy as np
v = np.arange(10,49)
print("Original vector:")
print(v)


# ### 8. Reverse a vector (first element becomes last)
# 

# In[62]:


import numpy as np

# Create a NumPy array (vector)
vector = np.array([1, 2, 3, 4, 5])

# Reverse the vector using slicing
reversed_vector = vector[::-1]

print("Original Vector:", vector)
print("Reversed Vector:", reversed_vector)


# ### 9. Create a 3x3 matrix with values ranging from 0 to 8
# 

# In[92]:


import numpy as np
x =  np.arange(0, 9).reshape(3,3)
print(x)


# ### 10. Find indices of non-zero elements from [1,2,0,0,4,0]
# 

# In[63]:


arr = np.array([1,2,0,0,4,0])
print(arr[0])
print(arr[1])
print(arr[4])


# ### 11. Create a 3x3 identity matrix (★☆☆)

# In[64]:


import numpy as np

# Create a 3x3 identity matrix
identity_matrix = np.identity(3)

print(identity_matrix)


# ### 12. Create a 3x3x3 array with random values (★☆☆)

# In[65]:


import numpy as np

# Create a 3x3x3 array with random values
random_array = np.random.rand(3, 3, 3)

print(random_array)


# ### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)

# In[66]:


import numpy as np

# Create a 10x10 array with random values between 0 and 1
random_array = np.random.rand(10, 10)

# Find the minimum and maximum values in the array
min_value = np.min(random_array)
max_value = np.max(random_array)

print("Random Array:")
print(random_array)
print("\nMinimum Value:", min_value)
print("Maximum Value:", max_value)


# ### 14. Create a random vector of size 30 and find the mean value (★☆☆)

# In[67]:


rv=np.random.random((30))
rv.mean()


# ### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)

# In[68]:


import numpy as np

# Define the shape of the 2D array
rows, cols = 5, 5  # Adjust the dimensions as needed

# Create a 2D array of zeros
array = np.zeros((rows, cols), dtype=int)

# Set the border elements to 1
array[0, :] = 1
array[-1, :] = 1
array[:, 0] = 1
array[:, -1] = 1

print(array)


# ### 16. How to add a border (filled with 0's) around an existing array? (★☆☆)

# In[69]:


import numpy as np

# Create an existing array (e.g., a 3x3 array)
existing_array = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])

# Determine the desired size for the new array with the border
rows, cols = existing_array.shape
border_size = 1  # Adjust the size of the border as needed

# Create a new larger array filled with 0's
new_array = np.zeros((rows + 2 * border_size, cols + 2 * border_size), dtype=existing_array.dtype)

# Copy the existing array into the center of the new array
new_array[border_size:border_size+rows, border_size:border_size+cols] = existing_array

print(new_array)


# ### 17. What is the result of the following expression? (★☆☆)

# In[70]:


print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(0.3 == 3 * 0.1)


# ### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)
# 

# In[71]:


mtrx=np.diag(np.arange(1,5),k=-1)
print(mtrx)


# ### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)

# In[72]:


import numpy as np

# Create an 8x8 matrix with a checkerboard pattern
checkerboard = np.zeros((8, 8), dtype=int)

# Set alternating rows and columns to 1
checkerboard[1::2, ::2] = 1  # Odd rows and even columns
checkerboard[::2, 1::2] = 1  # Even rows and odd columns

print(checkerboard)


# ### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?

# In[73]:


import numpy as np
print (np.unravel_index(100, (6,7,8)))


# ### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)
# 

# In[74]:


import numpy as np

# Create a 2x2 checkerboard pattern
checkerboard_pattern = np.array([[0, 1],
                                 [1, 0]])

# Use the tile function to replicate the pattern into an 8x8 matrix
checkerboard = np.tile(checkerboard_pattern, (4, 4))

print(checkerboard)


# ### 22. Normalize a 5x5 random matrix (★☆☆)
# 

# In[75]:


import numpy as np

# Create a 5x5 random matrix
random_matrix = np.random.rand(5, 5)

# Find the minimum and maximum values
min_value = random_matrix.min()
max_value = random_matrix.max()

# Normalize the matrix
normalized_matrix = (random_matrix - min_value) / (max_value - min_value)

print("Original Random Matrix:")
print(random_matrix)
print("\nNormalized Matrix:")
print(normalized_matrix)


# ### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)

# In[76]:


color = np.dtype([("r", np.ubyte,  (1,)),
                  ("g", np.ubyte,  (1,)),
                  ("b", np.ubyte,  (1,)),
                  ("a", np.ubyte,  (1,))])


# ### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)

# In[78]:


RMP = np.dot(np.ones((5,3)), np.ones((3,2)))
print(RMP)

# Alternative solution, in Python 3.5 and above
RMP = np.ones((5,3)) @ np.ones((3,2))
print(RMP)


# ### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)
# 

# In[79]:


X = np.arange(11)
X[(3 < X) & (X <= 8)] *= -1
print(X)


# ### 26. What is the output of the following script? (★☆☆)

# In[23]:


print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))


# ### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)
# 

# In[80]:


Z**Z
2 << Z >> 2
Z <- Z
1*Z
Z/1/1
Z<Z>Z


# ### 28. What are the result of the following expressions?
# 

# In[81]:


np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)


# ### 29. How to round away from zero a float array ? (★☆☆)
# 

# In[82]:


import numpy as np

# Create a float array
float_array = np.array([-1.5, 2.7, -3.4, 4.9, -5.2])

# Round away from zero using numpy.ceil() for positive numbers and numpy.floor() for negative numbers
rounded_array = np.where(float_array >= 0, np.ceil(float_array), np.floor(float_array))

print("Original Float Array:")
print(float_array)
print("\nRounded Array (Away from Zero):")
print(rounded_array)


# ### 30. How to find common values between two arrays? (★☆☆)
# 

# In[84]:


X1 = np.random.randint(0,11,10)
X2 = np.random.randint(0,11,10)
print(np.intersect1d(X1,X2))


# ### 31. How to ignore all numpy warnings (not recommended)? (★☆☆)

# In[30]:


with np.errstate(divide='ignore'):
    Z = np.ones(1) / 0


# ### 32. Is the following expressions true? (★☆☆)

# In[85]:


np.sqrt(-1) == np.emath.sqrt(-1)


# ### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)

# In[86]:


import numpy as np
from datetime import datetime, timedelta

# Get the current date (today)
today = datetime.now().date()

# Calculate the dates of yesterday and tomorrow
yesterday = today - timedelta(days=1)
tomorrow = today + timedelta(days=1)

# Create NumPy datetime64 objects
today_np = np.datetime64(today)
yesterday_np = np.datetime64(yesterday)
tomorrow_np = np.datetime64(tomorrow)

print("Yesterday:", yesterday_np)
print("Today:", today_np)
print("Tomorrow:", tomorrow_np)


# ### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)
# 

# In[88]:


import numpy as np
from datetime import datetime, timedelta

# Define the start date for July 2016
start_date = datetime(2016, 7, 1)

# Define the end date for July 2016
end_date = datetime(2016, 7, 31)

# Create an empty NumPy array to store the dates
dates_in_july_2016 = np.array([])

# Generate all the dates in July 2016
current_date = start_date
while current_date <= end_date:
    dates_in_july_2016 = np.append(dates_in_july_2016, current_date)
    current_date += timedelta(days=1)

# Convert the NumPy array to datetime64 data type
dates_in_july_2016 = np.array(dates_in_july_2016, dtype='datetime64')

# Print the array of dates
print(dates_in_july_2016)




# ### 35. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)
# 

# In[34]:


A = np.ones(3)*1
B = np.ones(3)*2
C = np.ones(3)*3
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(A,B,out=A)


# ### 36. Extract the integer part of a random array of positive numbers using 4 different methods (★★☆)
# 

# In[90]:


import numpy as np

# Create a random array of positive numbers
random_array = np.random.rand(5) * 100  # Generating random numbers between 0 and 100

# Method 1: Using numpy.floor()
integer_part_floor = np.floor(random_array).astype(int)

# Method 2: Using numpy.trunc()
integer_part_trunc = np.trunc(random_array).astype(int)

# Method 3: Using integer casting
integer_part_casting = random_array.astype(int)

# Method 4: Using numpy.floor_divide()
integer_part_floor_divide = np.floor_divide(random_array, 1).astype(int)

print("Original Random Array:")
print(random_array)
print("\nMethod 1 (numpy.floor()):", integer_part_floor)
print("Method 2 (numpy.trunc()):", integer_part_trunc)
print("Method 3 (integer casting):", integer_part_casting)
print("Method 4 (numpy.floor_divide()):", integer_part_floor_divide)


# ### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)
# 

# In[92]:


import numpy as np

# Create a 5x5 matrix with row values ranging from 0 to 4
matrix = np.tile(np.arange(5), (5, 1))

print(matrix)


# ### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)
# 

# In[93]:


import numpy as np

# Define a generator function that yields 10 integers
def integer_generator():
    for i in range(10):
        yield i

# Use the generator to build a NumPy array
integer_array = np.fromiter(integer_generator(), dtype=int)

print(integer_array)


# ### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)
# 

# In[94]:


import numpy as np

# Create a vector of size 10 with values ranging from 0 to 1 (both excluded)
vector = np.linspace(0, 1, 12)[1:-1]

print(vector)


# ### 40. Create a random vector of size 10 and sort it (★★☆)

# In[95]:


RV = np.random.random(10)
RV.sort()
print(RV)


# ### 41. How to sum a small array faster than np.sum? (★★☆)
# 

# In[96]:


SA = np.arange(9)
np.add.reduce(SA)


# ### 42. Consider two random array A and B, check if they are equal (★★☆)
# 

# In[97]:


import numpy as np

# Create two random arrays A and B
A = np.random.rand(5)
B = np.random.rand(5)

# Check if A and B are equal using numpy.array_equal()
are_equal = np.array_equal(A, B)

if are_equal:
    print("Arrays A and B are equal.")
else:
    print("Arrays A and B are not equal.")


# ### 43. Make an array immutable (read-only) (★★☆)
# 

# In[98]:


Z = np.zeros(10)
Z.flags.writeable = False
Z[0] = 1


# ### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)
# 

# In[99]:


import numpy as np

# Create a random 10x2 matrix of cartesian coordinates
cartesian_coordinates = np.random.rand(10, 2)

# Split the cartesian coordinates into x and y columns
x, y = cartesian_coordinates[:, 0], cartesian_coordinates[:, 1]

# Calculate the radial distance (r) using the Pythagorean theorem
r = np.sqrt(x**2 + y**2)

# Calculate the polar angle (theta) using arctan2
theta = np.arctan2(y, x)

# Convert radians to degrees if needed
# theta_degrees = np.degrees(theta)

# Create a 10x2 matrix of polar coordinates
polar_coordinates = np.column_stack((r, theta))

print("Cartesian Coordinates:")
print(cartesian_coordinates)
print("\nPolar Coordinates (r, theta):")
print(polar_coordinates)


# ### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)

# In[100]:


RV = np.random.random(10)
RV[RV.argmax()] = 0
print(RV)


# ### 46. Create a structured array with x and y coordinates covering the [0,1]x[0,1] area (★★☆)

# In[101]:


import numpy as np

# Define the size of the grid
grid_size = 5  # You can adjust the size as needed

# Create a grid of x and y coordinates covering the [0, 1] x [0, 1] area
x = np.linspace(0, 1, grid_size)
y = np.linspace(0, 1, grid_size)

x, y = np.meshgrid(x, y)

# Create a structured array
structured_array = np.empty((grid_size, grid_size), dtype=[('x', float), ('y', float)])
structured_array['x'] = x
structured_array['y'] = y

print(structured_array)


# ### 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))
# 

# In[102]:


import numpy as np

# Define the arrays X and Y
X = np.array([1, 2, 3, 4])
Y = np.array([0.5, 1.5, 2.5])

# Initialize the Cauchy matrix with zeros
C = np.zeros((len(X), len(Y)))

# Calculate the values for the Cauchy matrix
for i in range(len(X)):
    for j in range(len(Y)):
        C[i, j] = 1 / (X[i] - Y[j])

print("Cauchy Matrix C:")
print(C)



# ### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)

# In[103]:


import numpy as np

# Integer scalar types
integer_scalar_types = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]

print("Minimum and Maximum representable values for Integer Scalar Types:")
for dtype in integer_scalar_types:
    info = np.iinfo(dtype)
    print(f"{dtype.__name__}:")
    print(f"  Minimum: {info.min}")
    print(f"  Maximum: {info.max}")

# Floating-point scalar types
float_scalar_types = [np.float16, np.float32, np.float64]

print("\nMinimum and Maximum representable values for Floating-point Scalar Types:")
for dtype in float_scalar_types:
    info = np.finfo(dtype)
    print(f"{dtype.__name__}:")
    print(f"  Minimum: {info.min}")
    print(f"  Maximum: {info.max}")


# ### 49. How to print all the values of an array? (★★☆)
# 

# In[104]:


import numpy as np

# Create a NumPy array
array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Print all the values in the array
for value in array:
    print(value)


# ### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)

# In[105]:


import numpy as np

# Create a NumPy array
vector = np.array([3, 7, 1, 10, 5, 8])

# Scalar value to find the closest value to
scalar = 6

# Find the index of the closest value in the vector
closest_index = np.argmin(np.abs(vector - scalar))

# Get the closest value itself
closest_value = vector[closest_index]

print("Vector:", vector)
print("Scalar:", scalar)
print("Closest Value:", closest_value)


# ### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)

# In[106]:


SA = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print(SA)


# ### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)
# 

# In[107]:


import numpy as np

# Create a random vector with shape (100, 2) representing coordinates
coordinates = np.random.rand(100, 2)

# Calculate point-by-point distances
distances = np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2)

print("Point-by-Point Distances:")
print(distances)


# ### 53. How to convert a float (32 bits) array into an integer (32 bits) in place?
# 

# In[108]:


import numpy as np

# Create a float32 array
float_array = np.array([1.23, 4.56, 7.89], dtype=np.float32)

# Convert the float32 array to int32 in place
int_array = float_array.view(np.int32)

# Modify the integer array (if needed)
int_array[0] = 42  # Example: Modify the first element of the integer array

# Check the original float array and the modified integer array
print("Original Float Array (float32):", float_array)
print("Modified Integer Array (int32):", int_array)



# ### 54. How to read the following file? (★★☆)

# In[109]:


from io import StringIO

s = StringIO("""1, 2, 3, 4, 5\n
                6,  ,  , 7, 8\n
                 ,  , 9,10,11\n""")
Z = np.genfromtxt(s, delimiter=",", dtype=np.int32)
print(Z)


# ### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)
# 

# In[110]:


import numpy as np

# Create a NumPy array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Enumerate over the elements of the NumPy array along with their indices
for index, value in np.ndenumerate(arr):
    print(f"Index: {index}, Value: {value}")


# ### 56. Generate a generic 2D Gaussian-like array (★★☆)
# 

# In[113]:


import numpy as np

size = 5
x, y = np.meshgrid(np.linspace(0, size - 1, size), np.linspace(0, size - 1, size))
mean = (size - 1) / 2
sigma = 1.0
gaussian_array = np.exp(-((x - mean)**2 + (y - mean)**2) / (2 * sigma**2))

print(gaussian_array)



# ### 57. How to randomly place p elements in a 2D array? (★★☆)
# 

# In[114]:


import numpy as np

# Create a 2D array (e.g., a 5x5 array)
rows, cols = 5, 5
array = np.zeros((rows, cols))

# Number of elements to randomly place
p = 10

# Generate random row and column indices for placing elements
random_indices = np.random.choice(rows * cols, p, replace=False)
row_indices, col_indices = divmod(random_indices, cols)

# Place random values in the 2D array at the selected indices
for row, col in zip(row_indices, col_indices):
    array[row, col] = np.random.rand()  # Assign a random value

# Print the resulting 2D array
print(array)


# ### 58. Subtract the mean of each row of a matrix (★★☆)
# 

# In[116]:


J = np.random.rand(5, 15)

# Recent versions of numpy
K = J - J.mean(axis=1, keepdims=True)

# Older versions of numpy
L = J - J.mean(axis=1).reshape(-1, 1)

print(K)


# ### 59. How to sort an array by the nth column? (★★☆)

# In[117]:


nth = np.random.randint(0,10,(3,3))
print(nth)
print(nth[nth[:,1].argsort()])


# ### 60. How to tell if a given 2D array has null columns? (★★☆)
# 

# In[118]:


A = np.random.randint(0,3,(3,10))
print((~A.any(axis=0)).any())


# ### 61. Find the nearest value from a given value in an array (★★☆)
# 

# In[119]:


Z = np.random.uniform(0,1,10)
z = 0.5
m = Z.flat[np.abs(Z - z).argmin()]
print(m)


# ### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)

# In[120]:


import numpy as np

# Create two arrays with shape (1, 3) and (3, 1)
array1 = np.array([[1, 2, 3]])
array2 = np.array([[4], [5], [6]])

# Check if the arrays are compatible for element-wise addition
if array1.shape == (1, 3) and array2.shape == (3, 1):
    # Initialize the sum
    result = np.zeros((1, 1))

    # Create an iterator for array1
    it1 = np.nditer(array1)

    # Create an iterator for array2
    it2 = np.nditer(array2)

    # Iterate and add the elements
    while not it1.finished and not it2.finished:
        result += it1[0] + it2[0]
        it1.iternext()
        it2.iternext()

    print("Result of element-wise sum:", result[0, 0])
else:
    print("Array shapes are not compatible for element-wise addition.")


# ### 63. Create an array class that has a name attribute (★★☆)
# 

# In[121]:


import numpy as np

class NamedArray(np.ndarray):
    def __new__(cls, input_array, name=None):
        obj = np.asarray(input_array).view(cls)
        obj.name = name
        return obj

# Create a NamedArray with a name
arr = NamedArray([1, 2, 3], name="MyArray")

# Access the name attribute
print("Name:", arr.name)

# Access the array elements
print("Array:", arr)


# ### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)
# 

# In[122]:


import numpy as np

# Given vector
original_vector = np.array([1, 2, 3, 4, 5])

# Second vector representing indices to increment
indices_to_increment = np.array([1, 3, 3, 4, 4])

# Create a new vector with 1 added to the specified indices
result_vector = original_vector.copy()  # Copy the original vector to avoid modifying it in-place
unique_indices = np.unique(indices_to_increment)

# Iterate through the unique indices and increment the corresponding elements
for index in unique_indices:
    result_vector[index] += 1

print("Original Vector:", original_vector)
print("Indices to Increment:", indices_to_increment)
print("Result Vector:", result_vector)


# ### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)
# 

# In[123]:


import numpy as np

# Example vector X
X = np.array([1, 2, 3, 4, 5])

# Example index list I
I = np.array([1, 3, 0, 2, 4])

# Determine the size of the resulting array F
F_size = np.max(I) + 1

# Create an array F with zeros
F = np.zeros(F_size, dtype=X.dtype)

# Accumulate elements of X into F based on the index list I
for i in range(len(I)):
    F[I[i]] += X[i]

print("Vector X:", X)
print("Index List I:", I)
print("Accumulated Array F:", F)


# ### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★)

# In[124]:


import numpy as np

# Create a sample image as a NumPy array with shape (w, h, 3) and dtype 'ubyte'
image = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)

# Reshape the image to a (w*h, 3) 2D array
reshaped_image = image.reshape(-1, 3)

# Use numpy.unique to find unique colors
unique_colors = np.unique(reshaped_image, axis=0)

# Get the number of unique colors
num_unique_colors = unique_colors.shape[0]

print("Number of Unique Colors:", num_unique_colors)


# ### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)

# In[125]:


import numpy as np

# Create a sample 4D array
# Replace this with your own 4D array
four_dimensional_array = np.random.rand(2, 3, 4, 5)

# Get the sum over the last two axes (axis 2 and axis 3)
sum_over_last_two_axes = np.sum(four_dimensional_array, axis=(-2, -1))

print("Original 4D Array:")
print(four_dimensional_array)
print("Sum Over Last Two Axes:")
print(sum_over_last_two_axes)


# ### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset indices? (★★★)
# 

# In[126]:


import numpy as np

# Create a one-dimensional vector D and a corresponding vector S of subset indices
D = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
S = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1])  # Subset indices (0 or 1)

# Get unique subset indices
unique_indices = np.unique(S)

# Compute the means of subsets
subset_means = [np.mean(D[S == idx]) for idx in unique_indices]

# Print the means of subsets
for idx, mean in zip(unique_indices, subset_means):
    print(f"Subset {idx} Mean: {mean}")


# ### 69. How to get the diagonal of a dot product? (★★★)
# 

# In[127]:


import numpy as np

# Create two matrices A and B
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]])

# Compute the dot product of A and B
dot_product_result = np.dot(A, B)

# Get the diagonal of the dot product
diagonal_of_dot_product = np.diag(dot_product_result)

print("Matrix A:")
print(A)

print("Matrix B:")
print(B)

print("Dot Product of A and B:")
print(dot_product_result)

print("Diagonal of the Dot Product:")
print(diagonal_of_dot_product)


# ### 70. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)

# In[128]:


import numpy as np

# Original vector
original_vector = np.array([1, 2, 3, 4, 5])

# Number of consecutive zeros to insert
zeros_to_insert = 3

# Build a new vector with interleaved zeros
new_vector = np.zeros(len(original_vector) + (len(original_vector) - 1) * zeros_to_insert, dtype=original_vector.dtype)

# Assign values and zeros in an interleaved manner
new_vector[::zeros_to_insert + 1] = original_vector

print("Original Vector:")
print(original_vector)

print("New Vector with Interleaved Zeros:")
print(new_vector)


# ### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)

# In[129]:


import numpy as np

# Create a 3D array with dimensions (5, 5, 3)
array_3d = np.random.rand(5, 5, 3)

# Create a 2D array with dimensions (5, 5)
array_2d = np.random.rand(5, 5)

# Multiply the 3D array by the 2D array using broadcasting
result = array_3d * array_2d[:, :, np.newaxis]

print("3D Array:")
print(array_3d)

print("2D Array:")
print(array_2d)

print("Result (3D Array * 2D Array):")
print(result)


# ### 72. How to swap two rows of an array? (★★★)
# 

# In[131]:


import numpy as np

# Create a sample 2D array
array = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# Swap rows 1 and 2 (0-based indexing)
row1, row2 = 1, 2
array[row1], array[row2] = array[row2].copy(), array[row1].copy()

print("Original Array:")
print(array)


# ### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the triangles (★★★)

# In[132]:


# Sample set of 10 triplets describing triangles
triplets = [(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6),
            (5, 6, 7), (6, 7, 8), (7, 8, 9), (8, 9, 10),
            (9, 10, 1), (10, 1, 2)]

# Initialize a set to store unique line segments
unique_segments = set()

# Iterate through the triplets and extract line segments
for triplet in triplets:
    for i in range(3):
        segment = (triplet[i], triplet[(i + 1) % 3])
        unique_segments.add(segment)

# Convert the set of unique line segments to a list
unique_segments_list = list(unique_segments)

print("Unique Line Segments:")
for segment in unique_segments_list:
    print(segment)


# ### 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)

# In[133]:


import numpy as np

# Given bincount array C
C = np.array([0, 1, 2, 0, 1, 3, 2])

# Compute the array A
A = np.repeat(np.arange(len(C)), C)

print("Array A:", A)
print("Bincount of A:", np.bincount(A))


# ### 75. How to compute averages using a sliding window over an array? (★★★)
# 

# In[134]:


import numpy as np

# Sample array
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Define the window size for the sliding window
window_size = 3

# Compute averages using a sliding window
averages = np.convolve(arr, np.ones(window_size) / window_size, mode='valid')

print("Original Array:")
print(arr)
print(f"Averages with {window_size}-element sliding window:")
print(averages)


# ### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1]) (★★★)
# 

# In[135]:


import numpy as np

# Sample one-dimensional array Z
Z = np.array([1, 2, 3, 4, 5])

# Number of elements in each row
n = 3

# Build the two-dimensional array
result = np.lib.stride_tricks.sliding_window_view(Z, (n,))
result = result.T.copy()

print(result)


# ### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)

# In[136]:


# Original boolean value
boolean_value = True

# Negate the boolean in-place
boolean_value = not boolean_value

print("Negated Boolean:", boolean_value)


# ### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])? (★★★)

# In[137]:


import numpy as np

def distance_from_point_to_line(P0, P1, p):
    # Calculate the direction vector of the line
    v = P1 - P0

    # Calculate the vector from P0 to the point p
    w = p - P0

    # Calculate the projection of w onto v
    t = np.dot(w, v) / np.dot(v, v)

    # Calculate the closest point on the line to the point p
    closest_point = P0 + t * v

    # Calculate the distance from p to the closest point on the line
    distance = np.linalg.norm(p - closest_point)

    return distance

# Sample points and lines
P0 = np.array([[1, 2], [2, 3], [3, 4]])
P1 = np.array([[4, 5], [5, 6], [6, 7]])
p = np.array([2, 5])

# Calculate distances to each line
distances = [distance_from_point_to_line(P0[i], P1[i], p) for i in range(len(P0))]

print("Distances from point to lines:")
for i, distance in enumerate(distances):
    print(f"Line {i}: {distance}")


# ### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])? (★★★)

# In[139]:


import numpy as np

def distance_from_point_to_line(P0, P1, P):
    # Calculate the direction vector of the line
    v = P1 - P0

    # Calculate the vector from P0 to the point P
    w = P - P0

    # Calculate the projection of w onto v
    t = np.dot(w, v) / np.dot(v, v)

    # Calculate the closest point on the line to the point P
    closest_point = P0 + t * v

    # Calculate the distance from P to the closest point on the line
    distance = np.linalg.norm(P - closest_point, axis=1)

    return distance

# Sample points and lines
P0 = np.array([[1, 2], [2, 3], [3, 4]])
P1 = np.array([[4, 5], [5, 6], [6, 7]])
P = np.array([[2, 5], [3, 6], [4, 7]])

# Calculate distances from each point to each line
distances = np.array([distance_from_point_to_line(P0[i], P1[i], P) for i in range(len(P0))]).T

print("Distances from points to lines:")
print(distances)


# ### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a fill value when necessary) (★★★)

# In[141]:


import numpy as np

def extract_subpart(arr, center, shape, fill_value=0):
    subpart = np.full(shape, fill_value, dtype=arr.dtype)
    start = np.maximum(center - np.array(shape) // 2, 0)
    end = np.minimum(center + (np.array(shape) + 1) // 2, arr.shape)
    subpart_start = np.maximum(np.array(shape) // 2 - center + start, 0)
    subpart[subpart_start[0]:subpart_start[0]+end[0]-start[0], subpart_start[1]:subpart_start[1]+end[1]-start[1]] = arr[start[0]:end[0], start[1]:end[1]]
    return subpart

# Sample input array
input_array = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

# Center point for subpart
center_point = np.array([1, 1])

# Shape of the subpart
subpart_shape = (3, 3)

# Fill value for padding
fill_value = 0

# Extract subpart
subpart = extract_subpart(input_array, center_point, subpart_shape, fill_value)

print("Original Array:")
print(input_array)
print("Subpart:")
print(subpart)



# ### 81. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)

# In[142]:


import numpy as np

# Given array Z
Z = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

# Define the size of the rolling window
window_size = 4

# Generate the array R using a rolling window
R = np.lib.stride_tricks.sliding_window_view(Z, (window_size,))

print(R)


# ### 82. Compute a matrix rank (★★★)
# 

# In[144]:


W = np.random.uniform(0,1,(10,10))
X, Y, Z = np.linalg.svd(W) # Singular Value Decomposition
rank = np.sum(S > 1e-10)
print(rank)


# ### 83. How to find the most frequent value in an array?

# In[145]:


import numpy as np

# Sample array
array = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5])

# Use np.bincount to count occurrences of each value
counts = np.bincount(array)

# Find the value with the maximum count
most_frequent_value = np.argmax(counts)

print("Array:")
print(array)
print("Most frequent value:", most_frequent_value)


# ### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)
# 

# In[146]:


import numpy as np

# Create a random 10x10 matrix for demonstration
matrix = np.random.rand(10, 10)

# Define the block size
block_size = (3, 3)

# Initialize an empty list to store the extracted blocks
blocks = []

# Iterate through the matrix to extract blocks
for i in range(10 - block_size[0] + 1):
    for j in range(10 - block_size[1] + 1):
        block = matrix[i:i + block_size[0], j:j + block_size[1]]
        blocks.append(block)

# Convert the list of blocks to a NumPy array
block_array = np.array(blocks)

# Print the extracted blocks
for i, block in enumerate(block_array):
    print(f"Block {i + 1}:\n{block}\n")


# ### 85. Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★)
# 

# In[147]:


import numpy as np

class SymmetricArray(np.ndarray):
    def __new__(cls, input_array):
        # Create a view of the input array
        obj = np.asarray(input_array).view(cls)
        return obj

    def __getitem__(self, indices):
        # Ensure that Z[i, j] is equal to Z[j, i]
        i, j = indices
        return super(SymmetricArray, self).__getitem__((i, j))

    def __setitem__(self, indices, value):
        # Ensure that Z[i, j] is equal to Z[j, i] when setting values
        i, j = indices
        super(SymmetricArray, self).__setitem__((i, j), value)
        super(SymmetricArray, self).__setitem__((j, i), value)

# Create a regular NumPy array
original_array = np.array([[1, 2, 3],
                          [2, 4, 5],
                          [3, 5, 6]])

# Create a symmetric array using the custom subclass
symmetric_array = SymmetricArray(original_array)

# Access and modify elements
print("Original Array:")
print(original_array)
print("Symmetric Array:")
print(symmetric_array)

symmetric_array[0, 2] = 9
print("Modified Symmetric Array:")
print(symmetric_array)


# ### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)

# In[148]:


Q, n = 10, 20
R = np.ones((Q,n,n))
S = np.ones((Q,n,1))
T = np.tensordot(R, S, axes=[[0, 2], [0, 1]])
print(T)


# ### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)

# In[149]:


import numpy as np

# Create a random 16x16 array for demonstration
array = np.random.rand(16, 16)

# Define the block size
block_size = 4

# Calculate the number of blocks in each dimension
num_blocks = array.shape[0] // block_size

# Initialize an empty array to store the block-sums
block_sums = np.empty((num_blocks, num_blocks))

# Iterate through the blocks and calculate the block-sums
for i in range(num_blocks):
    for j in range(num_blocks):
        block = array[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
        block_sums[i, j] = np.sum(block)

# Print the block-sums
print("Block-Sums:")
print(block_sums)


# ### 88. How to implement the Game of Life using numpy arrays? (★★★)
# 

# In[150]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update(grid):
    # Copy the grid to calculate the next generation
    new_grid = grid.copy()
    rows, cols = grid.shape

    for i in range(rows):
        for j in range(cols):
            # Count the live neighbors
            neighbor_sum = np.sum(grid[i - 1:i + 2, j - 1:j + 2]) - grid[i, j]

            # Apply the Game of Life rules
            if grid[i, j] == 1:  # Cell is alive
                if neighbor_sum < 2 or neighbor_sum > 3:
                    new_grid[i, j] = 0
            else:  # Cell is dead
                if neighbor_sum == 3:
                    new_grid[i, j] = 1

    return new_grid

# Initialize a random grid
grid = np.random.choice([0, 1], size=(50, 50))

fig, ax = plt.subplots()
im = ax.imshow(grid, cmap='binary')

def animate(frame):
    global grid
    grid = update(grid)
    im.set_data(grid)
    return im,

ani = animation.FuncAnimation(fig, animate, frames=100, blit=True)
plt.show()


# ### 89. How to get the n largest values of an array (★★★)

# In[151]:


import numpy as np

# Create a NumPy array for demonstration
arr = np.array([4, 9, 1, 7, 3, 8, 5, 2, 6])

# Specify the number of largest values to retrieve
n = 3

# Use numpy.partition to partially sort the array
partitioned_indices = np.argpartition(-arr, n)  # Use negative values for largest elements

# Get the indices of the n largest values
n_largest_indices = partitioned_indices[:n]

# Get the n largest values from the original array
n_largest_values = arr[n_largest_indices]

print("Original Array:", arr)
print(f"{n} Largest Values:", n_largest_values)


# ### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)

# In[101]:


def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

print (cartesian(([1, 2, 3], [4, 5], [6, 7])))


# ### 91. How to create a record array from a regular array? (★★★)

# In[164]:


Z = np.array([("Hello", 2.5, 3),
              ("World", 3.6, 2)])
R = np.core.records.fromarrays(Z.T, 
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')
print(R)


# ### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)

# In[165]:


import numpy as np

# Create a large vector Z (replace with your own data)
Z = np.random.rand(1000000)  # Example vector with 1 million elements

# Method 1: Using NumPy's power function
result1 = np.power(Z, 3)

# Method 2: Using the ** operator
result2 = Z ** 3

# Method 3: Using NumPy's multiply function
result3 = np.multiply(np.multiply(Z, Z), Z)

# Verify if all methods produce the same result
assert np.allclose(result1, result2) and np.allclose(result2, result3)

# Print a portion of the result for verification
print("Method 1 Result (Sample):", result1[:10])




# ### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)

# In[166]:


import numpy as np

# Create two example arrays A and B
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [3, 2, 1],
              [9, 8, 7],
              [6, 5, 4],
              [2, 1, 3],
              [0, 0, 0]])

B = np.array([[1, 2],
              [7, 8]])

# Find the sorted unique elements in B
sorted_unique_B = np.sort(B, axis=1)
sorted_unique_B = np.unique(sorted_unique_B, axis=0)

# Find rows in A that contain elements of each row in sorted_unique_B
matching_rows = np.all(np.isin(A, sorted_unique_B), axis=1)

# Extract and print the matching rows from A
result = A[matching_rows]

print("Array A:")
print(A)
print("\nArray B:")
print(B)
print("\nRows of A that contain elements of each row of B:")
print(result)


# ### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) (★★★)
# 

# In[167]:


import numpy as np

# Create a 10x3 matrix (replace with your own data)
matrix = np.array([[2, 2, 2],
                  [1, 2, 3],
                  [3, 3, 3],
                  [4, 4, 4],
                  [1, 1, 1],
                  [2, 2, 2],
                  [5, 6, 5],
                  [7, 8, 9],
                  [0, 1, 2],
                  [3, 3, 3]])

# Check if each element in a row is not equal to the first element
unequal_rows = ~np.all(matrix == matrix[:, 0][:, np.newaxis], axis=1)

# Extract rows with unequal values
result = matrix[unequal_rows]

print("Original Matrix:")
print(matrix)
print("\nRows with Unequal Values:")
print(result)

  


# ### 95. Convert a vector of ints into a matrix binary representation (★★★)

# In[169]:


I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
print(B[:,::-1])

# Author: Daniel T. McDonald

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1))


# ### 96. Given a two dimensional array, how to extract unique rows? (★★★)

# In[170]:


import numpy as np

# Create a two-dimensional array (replace with your own data)
array_2d = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [1, 2, 3],
                    [7, 8, 9],
                    [4, 5, 6]])

# Use numpy.unique with the axis parameter to get unique rows
unique_rows = np.unique(array_2d, axis=0)

print("Original 2D Array:")
print(array_2d)
print("\nUnique Rows:")
print(unique_rows)


# ### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)

# In[171]:


A = np.random.uniform(0,1,10)
B = np.random.uniform(0,1,10)

np.einsum('i->', A)       # np.sum(A)
np.einsum('i,i->i', A, B) # A * B
np.einsum('i,i', A, B)    # np.inner(A, B)
np.einsum('i,j->ij', A, B)    # np.outer(A, B)


# ### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?

# In[172]:


import numpy as np

# Create two vectors X and Y describing the path
X = np.array([0, 1, 3, 6, 9, 11, 12])
Y = np.array([0, 2, 3, 4, 3, 2, 0])

# Calculate the total path length
path_length = np.cumsum(np.sqrt(np.diff(X) ** 2 + np.diff(Y) ** 2))
path_length = np.insert(path_length, 0, 0)  # Insert a zero at the beginning

# Define the number of equidistant samples
num_samples = 20

# Linearly interpolate the path at equidistant intervals
sampled_indices = np.linspace(0, path_length[-1], num_samples)
sampled_X = np.interp(sampled_indices, path_length, X)
sampled_Y = np.interp(sampled_indices, path_length, Y)

# Print the sampled path
sampled_path = np.column_stack((sampled_X, sampled_Y))
print("Sampled Path:")
print(sampled_path)


# ### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)

# In[174]:


X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)
print(X[M])


# ### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)

# In[175]:


import numpy as np

# Example 1D array X (replace with your own data)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Number of bootstraps
num_bootstraps = 1000

# Number of samples in each bootstrap (same as the original dataset)
sample_size = len(X)

# Initialize an array to store bootstrapped means
bootstrap_means = np.empty(num_bootstraps)

# Perform bootstrapping
for i in range(num_bootstraps):
    # Resample the array with replacement
    resampled_data = np.random.choice(X, size=sample_size, replace=True)
    # Calculate the mean of the resampled data
    bootstrap_means[i] = np.mean(resampled_data)

# Calculate the 2.5th and 97.5th percentiles to get the 95% confidence interval
confidence_interval = np.percentile(bootstrap_means, [2.5, 97.5])

print("Original 1D Array X:")
print(X)
print("\n95% Confidence Interval for the Mean (Bootstrapped):")
print(confidence_interval)


# In[ ]:





# In[ ]:




