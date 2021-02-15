"""
For Reference
for Pycharm :
https://www.geeksforgeeks.org/using-matplotlib-for-animations/
and
for Jupyter-Notebook :
http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-notebooks/
"""

import matplotlib.pyplot as plt
from matplotlib import animation, rc

def animated_plot(X,w_data):

	# creating a blank window for the animation
	fig = plt.figure()

	# limit of y  hard coded analysing it from the plot of w
	axis = plt.axes(xlim=(0, X.shape[1]), ylim=(-1000, 2000))

	line, = axis.plot([], [], lw = 2)

	# animation function. This is called sequentially at specified intervals
	def animate(i):
		x = list(range(X.shape[1]))
		y = w_data[i,:]
		line.set_data(x, y)
		return line,

	# call the animator. blit=True means only re-draw the parts that have changed.
	anim = animation.FuncAnimation(fig, animate, frames=100, interval=50, blit=True)
	plt.show()

