import matplotlib.pyplot as plt
import numpy as np
#https://matplotlib.org/tutorials/intermediate/artists.html#sphx-glr-tutorials-intermediate-artists-py
# =============================================================================
# There are two types of Artists: primitives and containers. The primitives 
# represent the standard graphical objects we want to paint onto our canvas: 
# Line2D, Rectangle, Text, AxesImage, etc., and the containers are places to 
# put them (Axis, Axes and Figure). 
# =============================================================================
fig=plt.figure()
ax=fig.add_subplot(2,1,1) # two rows, one column, first plot

# create an Axes at an arbitrary location, simply use the add_axes() method 
# which takes a list of [left, bottom, width, height] values in 0-1 relative 
# figure coordinates:
fig2=plt.figure()
ax2=fig2.add_axes([0.15,0.1,0.7,0.3])
t=np.arange(0,1,0.01)
s=np.sin(2*np.pi*t)
line,=ax2.plot(t,s,color='blue',lw=2)
# axe.lines is a list of length one
ax2.lines[0]
line
# remove lines by calling list methods
line1=ax2.plot([3,4,5])
del ax2.lines[1]
ax2.lines.remove(line1) # either of these two methods works
# The Axes also has helper methods to configure and decorate the x-axis and 
# y-axis tick, tick labels and axis labels:
xtext = ax2.set_xlabel('my xdata') # returns a Text instance
ytext = ax2.set_ylabel('my ydata')
# an example:
fig = plt.figure()
fig.subplots_adjust(top=0.8)
ax1 = fig.add_subplot(211)
ax1.set_ylabel('volts')
ax1.set_title('a sine wave')

t = np.arange(0.0, 1.0, 0.01)
s = np.sin(2*np.pi*t)
line, = ax1.plot(t, s, color='blue', lw=2)

np.random.seed(19680801)

ax2 = fig.add_axes([0.15, 0.1, 0.7, 0.3])
n, bins, patches = ax2.hist(np.random.randn(1000), 50,
                            facecolor='yellow', edgecolor='yellow')
ax2.set_xlabel('time (s)')

#--------------------------Customizing your objects(重要)------------------------
a=line.get_alpha()
line.set_alpha(0.5*a) # multiply the current alpha by a half, its None type here so doesn't work
line.set(alpha=0.01,zorder=2) # set a number of properties at once
line.get_color() # to see some certain property of an Artist
# a handy way to inspect the Artist properties is to use the matplotlib.artist.getp() 
# function (simply getp() in pylab), which lists the properties and their values
import matplotlib
matplotlib.artist.getp(ax2)
plt.getp(ax2)
plt.getp(line)
#to get a list of settable Artist properties, call setp() fucntion with a list of lines
plt.setp(ax2)
#-------------------------------Object Containers------------------------------

#------------1.Figure container: the top level container Artist---------------
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_axes([0.1,0.1,0.7,0.3])
ax1
print(fig.axes)
# you should not insert or remove axes directly from the axes list, but rather 
# use the add_subplot() and add_axes() methods to insert, and the delaxes() 
# method to delete. However, you are free to iterate over the list of axes:
for ax in fig.axes:
    ax.grid(True)
# =============================================================================
# The figure also has its own text, lines, patches and images, which you can use to 
# add primitives directly. The default coordinate system for the Figure will simply 
# be in pixels (which is not usually what you want) but you can control this by setting 
# the transform property of the Artist you are adding to the figure.

# More useful is “figure coordinates” where (0, 0) is the bottom-left of the figure 
# and (1, 1) is the top-right of the figure which you can obtain by setting the 
# Artist transform to fig.transFigure:
# =============================================================================
import matplotlib.lines as lines
fig = plt.figure()
l1 = lines.Line2D([0,1],[0,1],transform=fig.transFigure,figure=fig)
l2 = lines.Line2D([0,1],[1,0],transform=fig.transFigure,figure=fig)
fig.lines.extend([l1,l2]) # add the lines to the figure's lines list
fig

#-------------------------------2.Axes container-------------------------------
# Like the Figure, Axes contains a Patch patch which is a Rectangle for Cartesian 
# coordinates and a Circle for polar coordinates; this patch determines the 
# shape, background and border of the plotting region:
fig = plt.figure()
ax = fig.add_subplot(111)
rect = ax.patch # a Rectangle instance
rect.set_facecolor('green')
fig
# plot returns a list of lines because you can pass in multiple x, y pairs to plot, 
# and we are unpacking the first element of the length one list into the line variable. 
x,y = np.random.rand(2,100) # 2 by 100 random samples from a uniform distribution over [0,1)
line, = ax.plot(x,y,'-',color='blue',linewidth=2)
print(ax.lines)
# Similarly, methods that create patches, like bar() creates a list of rectangles, will 
# add the patches to the Axes.patches list:
n, bins, rectangles = ax.hist(np.random.randn(1000),50,facecolor='yellow',normed=1)
rectangles
print(len(rectangles))
fig.show()
#-----------------------------------------------------------
# You should not add objects directly to the Axes.lines or Axes.patches lists 
# unless you know exactly what you are doing.You can, nonetheless, create objects
# yourself and add them directly to the Axes using helper methods like add_line() and add_patch().
fig = plt.figure()
ax = fig.add_subplot(111)
# create a rectangle instance
import matplotlib
rect = matplotlib.patches.Rectangle((1,1),width=5,height=12)
# by default the axes instance is None
print(rect.axes)
# and the transformation instance is set to the 'identity transform'
print(rect.get_transform())
# now we add the Rectangle to the Axes
ax.add_patch(rect)
# notice that the ax.add_patch method has set the axes instance
print(rect.axes)
# and the transformation has been set too
print(rect.get_transform())
# the default axes transformation is ax.transData
print(ax.transData)
# note that the xlimits of the Axes have not been changed
print(ax.get_xlim())
# but the data limits have been updated to encompass the rectangle
print(ax.dataLim.bounds)
# we can manually invoke the auto-scaling machinery
ax.autoscale_view()
# and now the xlim are updated to encompass the rectangle
print(ax.get_xlim())
# we have to manually force a figure draw
fig.show()
# =============================================================================
# There are many, many Axes helper methods for creating primitive Artists and 
# adding them to their respective containers.这里略。
# =============================================================================

#---------------------------3.Axis container----------------------------------
# In addition to all of these Artists, the Axes contains two important Artist 
# containers: the XAxis and YAxis, which handle the drawing of the ticks and labels.
fig, ax = plt.subplots() 
axis = ax.xaxis # XAxis instance
axis.get_ticklocs() # get the x-axis ticks' locations
axis.get_ticklabels() # get the x-axis tick labels
# get the major ticks,and there are twice as many ticklines as labels(both top and 
# bottom of axis)
axis.get_ticklines() 
axis.get_ticklines(minor=True) # get he minor ticks
# Here is a summary of some of the useful accessor methods of the ``Axis``
# (these have corresponding setters where useful, such as
# set_major_formatter)
#
# ======================  =========================================================
# Accessor method         Description
# ======================  =========================================================
# get_scale               The scale of the axis, e.g., 'log' or 'linear'
# get_view_interval       The interval instance of the axis view limits
# get_data_interval       The interval instance of the axis data limits
# get_gridlines           A list of grid lines for the Axis
# get_label               The axis label - a Text instance
# get_ticklabels          A list of Text instances - keyword minor=True|False
# get_ticklines           A list of Line2D instances - keyword minor=True|False
# get_ticklocs            A list of Tick locations - keyword minor=True|False
# get_major_locator       The matplotlib.ticker.Locator instance for major ticks
# get_major_formatter     The matplotlib.ticker.Formatter instance for major ticks
# get_minor_locator       The matplotlib.ticker.Locator instance for minor ticks
# get_minor_formatter     The matplotlib.ticker.Formatter instance for minor ticks
# get_major_ticks         A list of Tick instances for major ticks
# get_minor_ticks         A list of Tick instances for minor ticks
# grid                    Turn the grid on or off for the major or minor ticks
# ======================  =========================================================
# Here is an example which customizes the axes and tick properties:
fig = plt.figure()
rect = fig.patch # a Rectangle instance (figure和axes都有patch)
rect.set_facecolor('lightgoldenrodyellow')

ax1 = fig.add_axes([0.1,0.3,0.4,0.4])
rect = ax1.patch
rect.set_facecolor('lightslategrey')

ax1.xaxis.get_label().set_text('haha')

for label in ax1.xaxis.get_ticklabels():
    # label is a Text instance
    label.set_color('red')
    label.set_rotation(45)
    label.set_fontsize(16)

for line in ax1.yaxis.get_ticklines():
    # line is a Line2D instance
    line.set_color('green')
    line.set_markersize(25)
    line.set_markeredgewidth(3)

fig.show()

#--------------------------4. Tick containers---------------------------------
import matplotlib.ticker as ticker
np.random.seed(19680801)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(100*np.random.rand(20))

formatter = ticker.FormatStrFormatter('$%1.2f')
ax.yaxis.set_major_formatter(formatter)

for tick in ax.yaxis.get_major_ticks():
    tick.label1On = False # tick label on the left
    tick.label2On = True # tick label on the right 
    tick.tick1On = False # tick on the left
    tick.tick2On = True # tick on the right
    tick.label2.set_color('green') 