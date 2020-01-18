import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
objects = ('N', 'V', 'R', 'A', 'P')
y_pos = np.arange(len(objects)  )
performance = [1543,460,2165,96,10]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Jumlah')
plt.xlabel('Kelas')
plt.title('Distribusi Label')
for i, v in enumerate(performance):
    ax.text(i, v, "%d" %v, ha='center', va='center' , fontsize=13)
plt.savefig('distrib_plot.png')    
plt.show()