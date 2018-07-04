import matplotlib.pyplot as plt
import numpy as np
#https://matplotlib.org/tutorials/intermediate/legend_guide.html#sphx-glr-tutorials-intermediate-legend-guide-py
'''-----------------------------legend demo---------------------------------'''
# To make a legend for lines already exist on the axes
ax = plt.subplot() # plt.subplots()返回的是一个figure和一个subplot
ax.plot([1,2,3])
ax.legend(['A sample line'])
# To keep the label and the legend element together, specify the label at artist 
# creation or by calling set_label method on the artist:
# legend由(label:文字) 和 (key:线型的图示) 组成
line,=ax.plot([1,2,3],label='Inline label')
ax.legend()
line.set_label('Label via method')
ax.legend()
# For full control of which artists have a legend entry:
x=np.arange(0,5,0.2)
line1,line2,line3=plt.plot(x,x**2,'r--',x,x**3,'bs',x,x**4,'g^')
plt.legend([line1,line2,line3],['red','blue','green'],loc='best')
line1.set_label('red888')
line2.set_label('blue888')
line3.set_label('green888')
plt.legend(loc=0) # 0 stands for 'best' lcoation

#---------------------------用handles更新labels-------------------------------
ax=plt.subplot()
ax.plot([4,3,2],label='example')
# calling legend() with no arguments is equivalent to:
handles,labels=ax.get_legend_handles_labels() # handles这里就是一个Line2D对象
ax.legend(handles,labels)

line_up,=plt.plot([3,2,1],label='line_2')
line_down,=plt.plot([1,2,3],label='line_1')
plt.legend(handles=[line_up,line_down],labels=['line_up','line_down'])

#---------------------------Proxy artists-----------------------------------
# Not all handles can be turned into legend entries automatically, so it is often 
# necessary to create an artist which can. Legend handles don’t have to exists on 
# the Figure or Axes in order to be used.
import matplotlib.patches as mpatches
red_patch=mpatches.Patch(color='red',label='The red data')
plt.legend(handles=[red_patch])

import matplotlib.lines as mlines
fig,ax=plt.subplots()
blue_line=mlines.Line2D([1,2,3],[5,6,7],color='blue',marker='*',markersize=15,label='blue stars')
ax.add_line(blue_line)
ax.autoscale_view() # 自动转换xlim和ylim
ax.legend(handles=[blue_line])
fig

#-----------------------------Legend Location---------------------------------
# if you want legend located at the figure's top-right corner instead of the axes' corner
plt.legend(handles=[blue_line],bbox_to_anchor=(1,1),bbox_transform=plt.gcf().transFigure)
# more examples:
plt.subplot(211)
plt.plot([1,2,3],label='test1')
plt.plot([3,2,1],label='test2')
plt.legend(bbox_to_anchor=(0., 1.021,1,0.1), loc=3, #后面的两个1是box的宽度和高度，当mode为expand时有用
           ncol=2, mode='expand',borderaxespad=0) #borderaxespad是填充box和axes之间的距离，用fontsize度量

plt.subplot(223)
plt.plot([1, 2, 3], label="test1")
plt.plot([3, 2, 1], label="test2")
# Place a legend to the right of this smaller subplot.
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)# (1.02,1)是box左上角的坐标,因为loc是upper left

#-------------------------Multiple legends on the same Axes--------------------
# split legend entries to multiple legends
line1,=plt.plot([1,2,3],label='Line 1',linestyle='--')
line2,=plt.plot([3,2,1],label='Line 2',linewidth=4)
# create a legend for the first line
first_legend=plt.legend(handles=[line1],loc=1) # loc参数，1是右上角，2是左上角，3是左下角，4是右下角
# add the legend manually to the current Axes,相当于把第一个legend和axes变成一个整体
ax=plt.gca().add_artist(first_legend)
# create another legend for the second line
plt.legend(handles=[line2],loc=4)

#-------------------------Legend Handlers-------------------------------------
# default handler_map has a special tuple handler (HandlerTuple) which simply plots 
# the handles on top of one another for each item in the given tuple. The following 
# example demonstrates combining two legend keys on top of one another:
x=np.random.randn(10)
red_dot,=plt.plot(x,'ro',linestyle='-',markersize=15)
# put a white cross over some of the data
white_cross,=plt.plot(x[:5],'w+',markeredgewidth=3,markersize=15)
plt.legend([red_dot,(red_dot,white_cross)],['red dot','red cross'])
# 剩余内容略