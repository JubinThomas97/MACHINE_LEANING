'''Numpy is a genral_purpose array processing package'''

import numpy as np


my_list=[1,2,3,4,5]                     # Arrays should be of same datatype

arr=np.array(my_list)                    #it creates an array based on the data type
print(type(arr))
print(arr.shape)                         #as it is a 1D array it will show as (5,) 

#Multinested array
my_list1=[1,2,3,4,5]
my_list2=[5,6,7,8,9]
my_list3=[9,10,11,12,13]

final_list=np.array([my_list1,my_list2,my_list3])
print(final_list)
print(final_list.shape)

new_array=final_list.reshape(5,3)            #By this method we can reshape an existing array into multidimesional one
array_1=final_list.reshape(1,15)             # reshape into 1 row and 15 colum
print(new_array)
print(array_1)


#########  Indexing   ##########

#Accessing the array elements
arr=np.array([1,2,3,4,5,6,7,8,9])
print(arr[3])                       #indexing of 1D array
print(final_list[:,:])             #indexing of 2D array
print(final_list[0:2,:])
print(final_list[0:2,0:2])
print(final_list[1:3,3:5])          #or we can use print(final_list(1:,3:))
print(final_list[1,1:4])
print(final_list[1:,2:4])


######## #Arange and linspace ##############

array1=np.arange(0,11,5)            #works like normal range function
print(array1)

array2=np.linspace(0,120,50)            #starts from 0 ends at 120 with 50 values of equal differnce
print(array2)


####### Copy function and broadcasting ########
print(arr)

arr[4:]=100         #So after the 4th element the rest of the element will be 100 Its called broadcasting   
print(arr)


array1_1=np.array([0,1,2,3,4,5,6,7,8,9,10])
array11_1=array1_1.copy()                           #used to copy any array
array11_1[7:]=100
print(array11_1)
print(array1_1)

######### Some conditions used in exploratary data analysis ########

val=2
print(arr*2)            #multiply 2 by each elements
print(arr<val)
print(arr[arr<2])       # will only print numbers greater than 2


new_arr=np.ones(4)      #will print only 1 , to print onlt int np.one(4,dtype-int)
print(new_arr)

nwq=np.random.rand(4,4)         #will give a array of 4*4 with values in between 0 and 1
print(nwq)

nsw=np.random.randint(0,100,8)      #select 8 numbers   between 0,100
print(nsw)


arr4=np.identity(4)
print(arr4)

arr5=np.zeros((4,5),dtype=int)
print(arr5)

arr6=np.random.random((2,3))
print(arr6)

###################################### 3D MAtrix ######################################

arrr1=np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
arrr1
arrr1.shape
arrr1.ndim   #arrr1 is the object we made from numpy class and 



###################################### Itteration ######################################
sample_array=np.random.random((5,5))
sample_array
for i in np.nditer(sample_array):
    print(i)



###################################### BRoadcasting ######################################
a3=np.arange(9).reshape(3,3)
a4=np.arange(3).reshape(1,3)
print(a3)
print(a4)
a3+a4   #here the first array is 3x3 and 2nd array is 1x3 but it still got added to 1st row of 2nd array to each row of 1st array
#just make sure that the no.of elemets of row or column for both the arrays are same.
# rows=coulm, colum= colum , rows= rows


a5=np.arange(3).reshape(1,3)
a6=np.arange(3).reshape(3,1)
print(a5)
print(a6)
a5+a6




a7=np.arange(1).reshape(1,1)
a8=np.arange(20).reshape(4,5)
print(a7)
print(a8)
a7+a8












