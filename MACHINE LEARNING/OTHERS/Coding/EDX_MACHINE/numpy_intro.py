########################################### NUMPY ###########################################

'''
BY using numpy we are building or seeing a Neural network which has a set of inputs and weights such as to obtain a result/output

'''

import numpy as np

#for input matrix
x_inp_matrix=[1,5]
x=np.array(x_inp_matrix)
print(x)

#x+1
#x*2

np.zeros([4,5],dtype=int)     #gives a 2-D array of 0s with [row,column]
np.ones([4,5],dtype=int)

np.random.random([3,2])             #anywhere between 0-1 [row,column]

nwq=np.random.rand(4,4)         #will give a array of 4*4 with values in between 0 and 1
print(nwq)

nsw=np.random.randint(0,100,8)      #select 8 numbers   between 0,100


#  EX1) Write a function called randomization that takes as input a positive integer n, and returns A, a random n x 1 Numpy array.

def randomization(n):
    random_array=np.random.random([n,1])
    return random_array
a=randomization(4)
print(a)

#############

y=np.array([[1,2,3,4],[5,6,7,8]])
np.transpose(y)


'''
Traditionally, NumPy will perform elementwise multiplication.However, there will be times when one will want to perform
traditional algebraic matrix multiplication.
For this, numbpy has built in np.matmul.
Now let's recall that to be able to multiply two matrices, the number of columns in the first matrix must be equal to the number of 
rows in the second matrix '''

x=np.array([3,5])
y=np.array([[6,8,4],[4,8,12]])
b=np.matmul(x,y)            #here X's no.of column should be equal to Y's number of rows
print(b)


'''
Exponential create the elements in x elementwise.
What this will do is raise e to each of the elements and x.
'''

c=np.array([10,5])
d=np.exp(c)
print(d)
np.sin(c)
np.cos(c)
np.tanh(c)


###### EX2) Write a function called operations that takes as input two positive integers h and w,---
# makes two random matrices A and B, of size h x w, and returns A,B, and s, the sum of A and B.

def operations(h, w):
    A=np.random.random([h,w])
    B=np.random.random([h,w])
    s=A+B
    return (f"A= {A},\n\n B= {B},\n\n Sum(s)= {s}")
matri_sum=operations(2,3)
print(matri_sum)


###########
new_arr=np.array([4,2,9,-6,5,11,13])
new_arr.max()
new_arr.min()

## NORM = Sq root of squared sum of all the elements a = [1,2,3,4,5]
#The L2 norm for the above is : sqrt(1^2 + 2^2 + 3^2 + 4^2 + 5^2) = 7.416

new_arr    
np.linalg.norm(new_arr)

#### EX3)Write a function called norm that takes as input two Numpy column arrays A and B, adds them, and returns s, the L2 norm of their sum.

def norm(A, B):
    mat_A=np.random.random([A,1])
    mat_B=np.random.random([B,1])
    mat_sum=mat_A+mat_B
    norm_mat=np.linalg.norm(mat_sum)
    return mat_sum,norm_mat
a=norm(3,3)
print(a)
