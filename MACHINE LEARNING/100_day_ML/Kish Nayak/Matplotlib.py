################################################ MatplotLib Tutorial  ###################################################

'''
Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. 
It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter,
 wxPython, Qt, or GTK+.

Some of the major Pros of Matplotlib are:

Generally easy to get started for simple plots
Support for custom labels and texts
Great control of every element in a figure
High-quality output in many formats
Very customizable in general

'''

import matplotlib.pyplot as plt
import numpy as np


## Simple Examples

x=np.arange(0,10)
y=np.arange(11,21)

a=np.arange(40,50)
b=np.arange(50,60)

######  plotting using matplot  #####

#SCTTER

plt.scatter(x, y,c='g')             # x is the x-axis value,y is y-axis value,s is solid pixels,c is colour
plt.xlabel('X Axis')
plt.ylabel('Y Axis') 
plt.title('Graph in 2D')
# plt.savefig('Test.png')     saves the image
plt.show()                      # to excute the plot

#PLOT
y=x*x
plt.plot(x,y,'go-')               # plot x and y using green circle markers
plt.show()

y=x*x
plt.plot(x,y,'r--')               # plot x and y using dashed line style
plt.show()

y=x*x
plt.plot(x,y,'r+')               # ditto, but with red plusses
plt.show()

y=x*x*x
plt.plot(x,y,'g*-',linewidth=2,markersize=8)
plt.xlabel('X Axis')
plt.ylabel('Y Axis') 
plt.title('Graph in 2D')
plt.show()


######## Creating Subplots ########

plt.subplot(2,2,1)
plt.plot(x,y,'r--')
plt.subplot(2,2,2)
plt.plot(x,y,'g*--')
plt.subplot(2,2,3)
plt.plot(x,y,'bo')
plt.subplot(2,2,4)
plt.plot(x,y,'go')
plt.show()



x = np.arange(1,11) 
y = 3 * x + 5 
plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
plt.plot(x,y) 
plt.show()


# Compute the x and y coordinates for points on a sine curve 

x = np.arange(0, 4 * np.pi, 0.1) 
y = np.sin(x) 
plt.title("sine wave form") 

plt.plot(x, y) 
plt.show() 


#Subplot()

# Compute the x and y coordinates for points on sine and cosine curves 
x = np.arange(0, 5 * np.pi, 0.1) 
y_sin = np.sin(x) 
y_cos = np.cos(x)  
   
# Set up a subplot grid that has height 2 and width 1, 
# and set the first such subplot as active. 
plt.subplot(2, 1, 1)
   
# Make the first plot 
plt.plot(x, y_sin,'r--') 
plt.xlabel('X Values')
plt.ylabel('Y Sin Values')
plt.title('Sine')  
   
# Set the second subplot as active, and make the second plot. 
plt.subplot(2, 1, 2) 
plt.plot(x, y_cos,'g--') 
plt.xlabel('X Values')
plt.ylabel('Y Cos Values')    
plt.title('Cosine')  
   
plt.show()


######################## BAR PLOTS ########################

x = [2,8,10] 
y = [11,16,9]  

x2 = [3,9,11] 
y2 = [6,15,7] 
plt.bar(x, y) 
plt.bar(x2, y2, color = 'g') 
plt.title('Bar graph') 
plt.ylabel('Y axis') 
plt.xlabel('X axis')  

plt.show()

######################## Histograms ########################

a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27]) 
plt.hist(a) 
plt.title("histogram") 
plt.show()

######################## Pie Chart ########################
# Data to plot
labels = 'Python', 'C++', 'Ruby', 'Java'
sizes = [215, 130, 245, 210]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.4, 0, 0, 0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True)

plt.axis('equal')
plt.show()