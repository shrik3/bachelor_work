import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns

## QUANT TEST STATS.

AL_3=[(0.9645), (0.9630), (0.9645), (0.9640), (0.9635), (0.9625), (0.9625), (0.9615), (0.9620), (0.9620), (0.9625)]
SL_3=[1.0, 0.0249205340114431, 0.018073370479787593, 0.016614935866272766, 0.016057739052391457, 0.01573613552223178, 0.0155192401181706, 0.015369657080887028, 0.015287386410381062, 0.015223813619535545, 0.015186417860214652]
TL_3= [0.25, 0.24, 0.23, 0.22, 0.21, 0.2, 0.19, 0.18, 0.16999999999999998, 0.16]


AL_5 = [0.9820, 0.9485, 0.9520, 0.9550, 0.9600, 0.9585, 0.9585, 0.9590, 0.9580, 0.9585, 0.9560]
SL_5= [1.0, 0.03391890577901663, 0.025978024827407385, 0.024584319191002495, 0.023871260493306973, 0.02359576054192461, 0.02340128998800765, 0.023287848831556088, 0.023158201795611448, 0.023125790036625286, 0.023044760639159886]
TL_5= [0.24, 0.22999999999999998, 0.22, 0.21, 0.19999999999999998, 0.19, 0.18, 0.16999999999999998, 0.15999999999999998, 0.15]

x_index = [0,1,2,3,4,5,6,7,8,9,10]

sns.lineplot(x=x_index,y=AL_3,markers="",label="accuracy")
plt.show()

sns.lineplot(x=x_index[1:],y=SL_3[1:],markers="",label="sparse")
plt.show()

sns.lineplot(x=x_index,y=AL_5,markers="",label="accuracy")
plt.show()

sns.lineplot(x=x_index[1:],y=SL_5[1:],markers="",label="sparse")
plt.show()

#


# sns.lineplot(x_index,SL_3_1)
plt.show()










