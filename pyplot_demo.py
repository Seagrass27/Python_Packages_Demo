import matplotlib.pyplot as plt
import numpy as np
#https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py
#-----------------------------
#plt.ion()# turn on the interactive mode
#plt.ioff()# turn off the interactive mode, must call plt.show() to show the plot
#plt.show()
#----------------------------------
plt.ion()
plt.plot([1,2,3,4]) # if provide a single array or list to plot(),it's assumed to be the y values
plt.ylabel('some numbers')
# plot x versus y:
plt.plot([1,2,3,4],[1,4,9,16])

# formatting the style of your plot
# the third argument is the formatting string:a color string+line style string,default is 'b-' 
# which is solid blue line
plt.plot(range(1,5),[i**2 for i in range(1,5)],'ro')
plt.axis([0,6,0,20])# takes[xmin,xmax,ymin,ymax]

# plotting several lines with different format styles in one command
t=np.arange(0,5,0.2)
plt.plot(t,t,'r--',t,t**2,'bs',t,t**3,'g^')# red dashes,blue squares,green triangles

# matplotlib allows provide an object in which variables can be accessed by strings,e.g.DataFrame
data={'a':np.arange(50),
      'b':np.arange(50),
      'c':np.random.randint(0,50,50),
      'd':np.abs(np.random.randn(50))*100}
plt.scatter('a','b',c='c',s='d',data=data) #c stands for mapped color, s stands for shape(marker size)
plt.xlabel('entry a')
plt.ylabel('entry b')

'''
#plotting with categorical variables(not work)
names=['group_a','group_b','group_c']
values=[1,10,100]
plt.figure(1,figsize=(9,3)) # figure number is 1
plt.subplot(131)
plt.bar(names,values)
plt.subplot(132)
plt.suptitle('failed!')
'''
#----------------------Controlling line properties------------------------
# there are several ways to set line properties
x=np.arange(0,50)
y=np.arange(0,50)+np.random.randn(50)
x1=np.arange(0,50)
y1=np.arange(0,50)+np.random.randn(50)
# 1.use keyword args:
plt.plot(x,y,linewidth=2.0)
# 2.use the setter methods of a Line2D instance
#   plot() returns a list of Line2D objects
line,=plt.plot(x,y,'-') # unpacking the list
line.set_antialiased(False) #turn off antialising
# 3.use setp() command,set multiple properties on a list of lines
lines=plt.plot(x,y,x1,y1)
plt.setp(lines,color='r',linewidth=2.0) # use keyword args
plt.setp(lines,'color','g','linewidth',2.0) # use MATLAB style
# to get a list of settable line properties, call setp() fucntion with a list of lines
plt.setp(lines)

#----------------------Working with multiple figures and axes------------------
# pyplot and MATLAB, all plotting commands apply to current axes
# gca() returns the current axes(a matplotlib.axes.Axes instance)
# gcf() returns the current figure(a matplotlib.figure.Figure instance)
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)
t1=np.arange(0,5,0.1)
t2=np.arange(0,5,0.02)
plt.figure(1)
plt.subplot(211) # specifies numrows,numcols,fignum ranges from 1 to numrows*numcols
plt.plot(t1,f(t1),'bo',t2,f(t2),'k')

plt.subplot(212)
plt.plot(t2,np.cos(2*np.pi*t2),'r--')

# =============================================================================
# if you want to place an axes manually,i.e.not on a rectangular grid,use axes() command
# which allows you to specify the location as axes([left,bottom,width,height]) where values are fractional
# =============================================================================

plt.figure(1) # the first figure
plt.subplot(211) # the first subplot in the first figure
plt.plot([1,2,3])
plt.subplot(212) # the second subplot in the first figure
plt.plot([4,5,6])

plt.figure(2) # a second figure
plt.plot([4,5,6])

plt.figure(1) # figure 1 current;subplot(212) still current
plt.subplot(211) # make subplot(211) in figure1 current
plt.title('Easy as 1,2,3')
# you can clear the current figure with clf(),and the current axes with cla()
plt.cla()
plt.close(plt.figure(1)) # to release memory

#---------------------------Working with text---------------------------------
mu,sigma=100,15
x=mu+sigma*np.random.randn(10000)
n,bins,patches=plt.hist(x,50,normed=1,facecolor='g',alpha=0.75)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60,.025,r'$\mu=100,\ \sigma=15$')
# all of the text() command returns an matplotlib.text.Text instance,similar to lines.
t=plt.xlabel('mydata',fontsize=14,color='red')
plt.setp(t,color='g')
# using mathematical expressions in text
# mpl accepts TeX equation expressions in any text expression
plt.title(r'$\sigma_i=15$')# write TeX expression surrounded by dollar signs
# annotating text
ax=plt.subplot(111)
t=np.arange(0,5,0.01)
s=np.cos(2*np.pi*t)
plt.plot(t,s,lw=2)
plt.annotate('local max',xy=(2,1),xytext=(3,1.5),
             arrowprops=dict(facecolor='black',shrink=0.005))
plt.ylim(-2,2)

#---------------------Logarithmic and other nonlinear axes--------------------
from matplotlib.ticker import NullFormatter # useful for logit scale

# fixing random state for reproducibility
np.random.seed(19680801)

y=np.random.normal(0.5,0.4,1000)
y=y[(y>0) & (y<1)]
y.sort()
x=np.arange(len(y))
plt.figure(1)

# linear
plt.subplot(221)
plt.plot(x,y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)
# log
plt.subplot(222)
plt.plot(x,y)
plt.yscale('log')
plt.title('log')
plt.grid(True)
# symmtric log
plt.subplot(223)
plt.plot(x,y-y.mean())
plt.yscale('symlog',linthreshy=0.01)
plt.title('symlog')
plt.grid(True)
# logit
plt.subplot(224)
plt.plot(x,y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)
# Format the minor tick labels of the y-axis into empty strings with
# `NullFormatter`, to avoid cumbering the axis with too many labels.
plt.gca().yaxis.set_minor_formatter(NullFormatter())
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95, hspace=0.25,
                    wspace=0.35)

