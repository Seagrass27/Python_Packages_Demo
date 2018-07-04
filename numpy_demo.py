import numpy as np
demo=np.array([[0,1,2],[3,4,5]],dtype=np.float64)
demo
demo.ndim #dimensions of the array, known as rank in numpy
demo.shape #a tuple indicating the size of array
demo.size #number of elements in the array
demo.dtype #describing the type of the elements in the array
demo.itemsize #the size in bytes of each element of the array,here float64 is 8 bytes
#---------an example--------------------------------------------------
a = np.arange(24).reshape(2,3,4)
a
print(a)#the last axis which is 4 here is printed from left to right
        #the second-to-last axis which is 3 here is printed from top to bottom
        #the rest axes are also printed from top to bottom, but with one empty line seprated from each other
a.dtype.name
a.itemsize
type(a)
'''-------------Arrays creation--------------------------------------------'''
#Note:array function takes only one argument
#use list to create array with array function
np.array([[2,3,4],[3,4,5]])
#use tuple to create array with array function
np.array(((12,3,5),(3,4,6)))
#can specify the type of array when creating the array
np.array([[1,2,3],[2,3,4]],dtype=np.complex)
#function zeros creates an array full of zeros
np.zeros((3,4))
#function ones creates an array full of ones
np.ones((2,3))
#function empty creates an array whose initial content is random and dempends on 
#the states of memory'''
np.empty((2,3))
#arange creates a sequence of numbers, returns an array instead of list
np.arange(10,30,5)
np.arange(0,2,0.3)
#linspace receives the number of elements we want instead of step
np.linspace(0,2,9) 
x=np.linspace(0,2*np.pi,100)
np.sin(x)
'''------------------Basic operations---------------------------------------'''
# Arithmatic operators apply elementwise. A new array filled with results is created.
a=np.arange(4)
a**2
10*np.sin(a)
a<35
# matrix product can be done by dot function
a=np.array([[12,3],[4,5]])
b=np.array([[2,3],[5,6]])
a*b #elementwise product
np.dot(a,b) #matrix product
# +=, *= act in place to modify an existing array rather than create a new one
a=np.ones((2,3),dtype=np.int)
b=np.random.random((2,3))
a*=3 # equivalent to a=a*3
a
b+=a # equivalent to b=b+a
b
# upcasting:operating with arrays of different types, the result corresponds to 
# the more general or precise one
a=np.ones(3,dtype=np.int)
b=np.linspace(0,np.pi,3)
c=a+b
c.dtype.name
d=np.exp(c*1j)
d.dtype.name
# unary operations are implemented as methods of the ndarray class
a=np.random.random((2,3))
# treat the elements as a list of numbers regardless of its shape
a.sum()
a.min()
a.max()
# specifying the axis parameter 
b=np.arange(12).reshape(3,4)
b
b.sum(axis=0)#sum of each column
b.min(axis=1)#min of each row
b.cumsum(axis=1)#cumulative sum along each column
'''-----------------------Universal functions------------------------------'''
# universal functions operate elementwise, known as 'ufunc',producing an new array
b=np.arange(3)
np.exp(b)
np.sqrt(b)
c=np.array([2,-1.,4])
np.add(b,c)
'''----------------------Indexing,Slicing and Iterating---------------------'''
# one-dimensional arrays can be indexed, sliced, and iterated over 
a=np.arange(10)**3
a
a[2]
a[2:5]
a[:6:2]=-1000 #equivalent to a[0:6:2]=-1000, position 6 is exclusive
a
a[::-1]
for i in a:
    print(i**(1/3))
# multidimentional arrays can have one index per axis, these indices are given 
# in a tuple seperated by commas
def f(x,y):
    return 10*x+y
b=np.fromfunction(f,(5,4),dtype=np.int)
b
b[2,3]
b[0:5,1] #each row in the second column of b
b[:,1] #equivalent to the previous example
b[1:3,:]
b[-1] #when fewer indices are povided than the number of axes, the missing indices
      #are considered complete slices, equivalent to b[-1,:]
# dots(...)represent as many colons as needed to produce a complete indexing tuple
# x[1,2,...] equivalent to x[1,2,:,:,:] here x is an array of rank 5
# x[1,...,5,:] equivalent to x[1,:,:,5,:]
c=np.array([[[1,5,9],[2,8,4]],[[3,11,6],[7,13,10]]])
c.shape
c[1,...]
c[...,2]
# Iterating over multidimensional arrays is done with respect to the first axis:
b
for row in b:
    print(row)
# if we want to iterate over each element, we can use flat attribute which is an 
# iterator over all the elements of the array:
for element in b.flat:
    print(element)
'''------------------------------Shape Manipulation-------------------------'''
a=np.floor(10*np.random.random((3,4)))
a.shape
a.ravel() #flatten the array using 'C style',returns a view of a
a.flatten() #flatten the array but returns a new array(copy)
a.shape=(6,2)
a.transpose()
# reshape function returns a new array but resize method modifies the array itself
a.resize((2,6))
a
a.reshape(3,-1) #if dimension is given as -1 in a reshaping operation, its automatically calculated
'''------------------------Stacking togethe different arrays----------------'''
a=np.floor(10*np.random.random((2,2)))
a
b=np.floor(10*np.random.random((2,2)))
b
np.vstack((a,b)) #vertical(其实沿着（along）的方向是行，因为行变,同理考虑之前的sum()中的axis)
np.hstack((a,b)) #horizonal
np.concatenate((a,b),axis=1)#concatenate can specify along which axis to stack
np.r_[a,b]#stack along row，which is first axis
np.c_[a,b]#stack along column, which is last axis
a=np.array([4,2])
a[:,np.newaxis] #this allows to have a 2D columns vector
'''--------------------Splitting array into several smaller ones------------'''
a=np.floor(10*np.random.random((2,12)))
np.hsplit(a,3) #horizontally split a into 3 arrays
np.hsplit(a,(3,4)) #split after the third and the fourth column
np.array_split(a,(3,4),axis=1) # same as previous example
'''--------------------Copies and Views-------------------------------------'''
#1.---No copy at all
a=np.arange(12)
b=a # no new object is created
b is a # b and a are two names for the same ndarray object
b.shape=3,4 # it also changes the shape of a
a.shape
# Python passes mutable objects as references, so function calls makes no copy
def f(x):
    print(id(x))
id(a)
f(a)
#2.---View or shallow copy
# different array objects can share the same data
c=a.view()
c is a
c.base is a #c is a view of the data owned by a
c.shape = 2,6 # a's shape doesn't change
a.shape
c[0,4]=1000 # a's data changes
a
# slicing an array returns a view of it
s=a[:,1:3]
s[:]=555
a
#3.---Deep copy
# the copy method makes a complete copy of the array and its data
d=a.copy() # a new array object with new data is created
d is a
d[0,0]=666 # a doesn't change 
a
'''--------------------Fancy Indexing and index tricks----------------------'''
# Indexing with arrays of indices
a=np.arange(12)**2
i=np.array([2,2,5,6]) 
a[i] #the elements of a at position i
j=np.array([[3,4],[9,7]])
a[j] #the same shape as j
# When indexed array a is multidimentional, a single array of indices refers to
# the first dimension of a:
palette=np.array([[0,0,0],        # black
                  [255,0,0],      # red
                  [0,255,0],      # green
                  [0,0,255],      # blue
                  [255,255,255]]) # white
image=np.array([[0,1,2,0],        # each value corresponds to a color
                [0,3,4,0]])
palette[image]
# We can also give indexes to more than one dimension. The arrays of indices for
# each dimension must have the same shape
a=np.arange(12).reshape(3,4)
a
i=np.array([[2,1],  # indices for the first dim
            [1,0]])
j=np.array([[2,0],  # indices for the second dim
            [1,3]])
a[i,j]              # i and j must have the same shape
a[i,0]              
a[:,j]
l=[i,j] # usually we put i and j in a sequence(list here) and indexing with the list
a[l]
# indexing with arrays can be used on searching for the maximum value:
time=np.linspace(0,30,5)
data=np.sin(np.arange(20).reshape(5,4))
time
data
ind=data.argmax(axis=0) # the postion of maximum value in each colum
ind
time_max=time[ind] # times corresponding to the maxima
data_max=data[ind,np.arange(data.shape[1])]
np.all(data_max==data.max(axis=0))
# we can also use indexing with arrays as a target to assign to:
a=np.arange(5)
a
a[[0,1,2]]=88 # note this is actually indexing with 'list' rather than array
a

#---Indexing with Boolean Arrays
# 1.use boolean arrays that have the same shape as the original array
a=np.arange(15).reshape(3,5)
b=a>4
b # b is a boolean with a's shape
a[b] # 1d array with selected elements
a[b]=0 # all elements of 'a' higher than 4 become 0
a
# 2.for each dimension, we give a 1D boolean array
a=np.arange(12).reshape(3,4)
b1=np.array([False,True,True]) # of length 3 matching the number of rows
b2=np.array([True,False,True,False]) # of length 4 matching the number or columns
a
a[b1,:] #selecting rows
a[b1] #same thing
a[:,b2] #selecting columns
a[b1,b2] # a weidrd thing to do

#---The ix_ function
# if you want to compute all the a+b*c for all the triplets taken from each of the
# vectors a, b, and c:
a=np.array([2,3,4,5])
b=np.array([8,5,4])
c=np.array([5,4,6,8,3])
ax,bx,cx=np.ix_(a,b,c)
ax
bx
cx
ax.shape,bx.shape,cx.shape
result=ax+bx*cx
result
result[3,2,1]==a[3]+b[2]*c[1]
'''---------------------------Linear Algebra--------------------------------'''
a=np.array([[3,4],[2,1]],dtype=np.float64)
a.T           #a's transpose
a.transpose() #same thing
np.linalg.inv(a) # inverse of a
np.dot(a,np.linalg.inv(a)) # matrix product
u=np.eye(2) # identity matrix 
np.trace(u) # trace
y=np.array([5,7])
x=np.linalg.solve(a,y) # solve linear equation
np.dot(a,x) 
result=np.linalg.eig(a) #returns eigen values, and the corresponding eigen vectors(each column)
'''------------------------The matrix class--------------------------------'''
A=np.matrix('1 2; 3 4')
A
type(A) # file where class is defined
A.T    # transpose
X=np.matrix('5 7')
Y=X.T
Y
print(A*Y) # matrix product
print(A.I) # matrix inverse
np.linalg.solve(A,Y) # solve linear equation
#---Indexing matrices
A=np.arange(12)
A
A.shape=3,4
M=np.mat(A.copy())
print(type(A),' ',type(M))
print(A)
print(M)
# A slice of matrix will always produce a matrix
# A slice of an array will always produce an array in of the lowest possible dimension
print(A[:,1]);print(A[:,1].shape)  # it's a 1-dimensional array
print(M[:,1]);print(A[:,1].shape)  # it's a 2-dimensional matrix
A[:,[0,2]] # the 0th and 2rd columns 
A[:,].take([0,2],axis=1) # a more complicated way to to it
A[1:,].take([0,2],axis=1) # skip the first row
A[1:,[0,2]] # the same thing
A[np.ix_([1,2],[0,2])] # the cross-product way to do it
# to get all the columns where the first row is greater than 1
A[0,:]>1
A[:,A[0,:]>1]
M[:,M.A[0,:]>1] # use A attribute whose value is the array representation of that matrix
# to conditionally slice the matrix in two directions, we need use ix_
A[np.ix_(A[:,0]>2,A[0,:]>1)]
M[np.ix_(M.A[:,0]>2,M.A[0,:]>1)]

'''-------------------------Histograms-------------------------------------'''
import pylab
mu,sigma=2,0.5
v=np.random.normal(mu,sigma,100)
# hist function built in matplotlib, plots the histograms automatically
pylab.hist(v,bins=10,normed=1)
# histogram function built in numpy, only generates the data
n,bins=np.histogram(v,bins=10,normed=True)
n,bins # bins是11个取值点，n是占比（normed=False的话是个数）
pylab.plot(.5*(bins[1:]+bins[:-1]), n) #变成10个点

'''-----------------------np.random---------------------------------------'''
np.random.rand(2,3) # 2*3 的array，从[0,1)的均匀分布中sample
np.random.randn(2,3)# 2*3 的array, 从标准正态分布中sample
np.random.random((2,3)) # 2*3 的array，从[0,1)的均匀分布中sample,参数是tuple
np.random.randint(0,100,50) # 50(接受tuple)个[0,100)中随机取的整数 
np.random.normal(0,1,100) # 100(接受tuple)个均值为1，标准差为1的正态分布
