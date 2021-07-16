import seaborn as sns
import matplotlib.pyplot as plt



sns.set_style("whitegrid")
tips = sns.load_dataset("tips") #载入自带数据集
#x轴为分类变量day,y轴为数值变量total_bill，利用颜色再对sex分类
ax = sns.barplot(x="day", y="total_bill", hue="sex", data=tips) 
plt.show()
