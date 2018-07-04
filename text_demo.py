#https://matplotlib.org/tutorials/text/text_intro.html#sphx-glr-tutorials-text-text-intro-py
'''The following commands are used to create text in the pyplot interface
text() 
- add text at an arbitrary location to the Axes; matplotlib.axes.Axes.text() in the API.
xlabel() 
- add a label to the x-axis; matplotlib.axes.Axes.set_xlabel() in the API.
ylabel() 
- add a label to the y-axis; matplotlib.axes.Axes.set_ylabel() in the API.
title() 
- add a title to the Axes; matplotlib.axes.Axes.set_title() in the API.
figtext() 
- add text at an arbitrary location to the Figure; matplotlib.figure.Figure.text() in the API.
suptitle()
- add a title to the Figure; matplotlib.figure.Figure.suptitle() in the API.
annotate() 
- add an annotation, with
optional arrow, to the Axes ; matplotlib.axes.Axes.annotate() in the API.
'''
import matplotlib.pyplot as plt
fig = plt.figure()
fig.suptitle('bold figure supertitle',fontsize=14,fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('axes title')

ax.set_xlabel('xlabel')
ax.set_ylabel('ylabel')

ax.axis([0,10,0,10])

ax.text(3,8,'boxed italics text in data coords',style='italic',
        bbox={'facecolor':'red','alpha':0.5,'pad':10})

ax.text(2,6,r'an equation: $E=mc^2$',fontsize=15)

ax.text(0.95,0.01,'colored text in axes coords',
        verticalalignment='bottom',horizontalalignment='right',
        transform=ax.transAxes,
        color='green',fontsize=15)

ax.plot([2],[1],'o')
ax.annotate('annotate',xy=(2,1),xytext=(3,4),
            arrowprops=dict(facecolor='black',shrink=0.05))