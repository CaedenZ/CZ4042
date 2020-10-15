import numpy as np
import matplotlib.pyplot as plt
        
def show(X):
    if X.ndim == 2:
	    plt.imshow(X, cmap='gray')
	    plt.show()
    elif X.ndim == 3:
        plt.imshow(X)
        plt.show()
    else:
        print('WRONG TENSOR SIZE')

def show_prob_mnist(p):
	ft=15
	label = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight','nine')
	y_pos = np.arange(len(p))*1.2
	target=2
	width=0.9
	col= 'blue'
	#col='darkgreen'

	plt.rcdefaults()
	fig, ax = plt.subplots()

	# the plot
	ax.barh(y_pos, p, width , align='center', color=col)

	ax.set_xlim([0, 1.3])

	# y label
	ax.set_yticks(y_pos)
	ax.set_yticklabels(label, fontsize=ft)
	ax.invert_yaxis()  

	# x label
	ax.set_xticklabels([])
	ax.set_xticks([])

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['left'].set_linewidth(4)


	for i in range(len(p)):
	    str_nb="{0:.2f}".format(p[i])
	    ax.text( p[i] + 0.05 , y_pos[i] ,str_nb ,
	             horizontalalignment='left', verticalalignment='center',
	             transform=ax.transData, color= col,fontsize=ft)

	plt.show()
	# fig.savefig('./figures/prob', dpi=96, bbox_inches="tight")
