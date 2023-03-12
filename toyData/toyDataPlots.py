import numpy as np
import matplotlib.pyplot as plt
from toyDataAA import toyDataAA
from scipy.spatial import ConvexHull, convex_hull_plot_2d

#run the AA based on number of archetypes
numArchetypes=[3,5,7,11]
#numArchetypes=[3]
archeTypesList=[]
losses=[]
numIterations=10000
for numArchetype in numArchetypes:
    data, archeTypes, loss_Adam = toyDataAA(numArchetypes=numArchetype,numIterations=numIterations)
    archeTypesList.append(archeTypes)
    losses.append(loss_Adam)

fig1, ax1 = plt.subplots()
#plot the loss functions
for idx,loss in enumerate(losses): 
    ax1.plot(range(1,numIterations+1), loss, label=f'Loss with {numArchetypes[idx]} archetypes. Final loss: {loss[-1]:.2f}')

ax1.set_title('Loss Comparison')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')
ax1.set_yscale('log')
ax1.legend()
fig1.savefig('toyData/plots/lossComparison.png')

#plot the data together with the found archetypes. 
distributions=[data[:,0:100],data[:,100:200],data[:,200:]]

fig2, ax2 = plt.subplots()

matplotlibStandardcolors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

for idx,archeTypes in enumerate(archeTypesList):    
    points=archeTypes.T    
    hull = ConvexHull(points)    
    for simplex in hull.simplices:        
        ax2.plot(archeTypes[0,:], archeTypes[1,:], 'x', color=matplotlibStandardcolors[idx], alpha=1,label=f'Convex hull with {idx} archetypes')
        ax2.plot(points[simplex, 0], points[simplex, 1],'k-',color=matplotlibStandardcolors[idx])   
oldIdx=idx
for idx,distribution in enumerate(distributions,oldIdx+1):
    ax2.plot(distribution[0,:], distribution[1,:], '.', color=matplotlibStandardcolors[idx], alpha=0.5)

ax1.set_title('Archetypes and their convex hull')
ax2.legend()

fig1.savefig('toyData/plots/convexHulss.png')   
plt.show()

