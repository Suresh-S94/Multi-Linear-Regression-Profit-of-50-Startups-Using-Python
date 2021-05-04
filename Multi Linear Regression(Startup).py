import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import numpy as np

startup = pd.read_csv("E:/Data Science_ExcelR/Multi Linear Regression/50_Startups.csv")
startup.head()

startup.info()

startup.isna().sum()

startup.corr()


sns.set_style(style='darkgrid')
sns.pairplot(startup)

plt.boxplot(startup["Profit"])
plt.boxplot(startup["R&D Spend"])
plt.boxplot(startup["Administration"])
plt.boxplot(startup["Marketing Spend"])

Startup= pd.get_dummies(startup['State'])

startup= pd.concat([startup,Startup],axis=1)

startup= startup.drop(["State"],axis=1)

startup= startup.iloc[:,[3,0,1,2,4,5,6]]

sns.pairplot(startup)
cor_values= startup.corr()


from sklearn.model_selection import train_test_split

train_data,test_data= train_test_split(startup)

train_data.to_csv("train_data.csv",encoding="utf-8")
test_data.to_csv("test_data.csv",encoding="utf-8")

startup.rename(columns={'R&D Spend': 'RnD','Marketing Spend':'Marketing','New York':'NewYork'},inplace=True)
train_data.rename(columns={'R&D Spend': 'RnD','Marketing Spend':'Marketing','New York':'NewYork'},inplace=True)
test_data.rename(columns={'R&D Spend': 'RnD','Marketing Spend':'Marketing','New York':'NewYork'},inplace=True)

model1= smf.ols("Profit~RnD+Administration+Marketing+California+Florida+NewYork",data=train_data).fit()
model1.summary()

model1_ad= smf.ols("Profit~Administration", data= train_data).fit()
model1_ad.summary()


model1_ma = smf.ols("Profit~Marketing", data= train_data).fit()
model1_ma.summary()

model1_com= smf.ols("Profit~Administration+Marketing", data= train_data).fit()
model1_com.summary()

import statsmodels.api as sm
sm.graphics.influence_plot(model1)

train_data1= train_data.drop(train_data.index[[4]], axis=0)

model2= smf.ols("Profit~RnD+Administration+Marketing+California+Florida+NewYork", data= train_data1).fit()
model2.summary()

train_data2 = train_data.drop(train_data.index[[4,24]],axis=0)
model3 = smf.ols("Profit~RnD+Administration+Marketing+California+Florida+NewYork", data= train_data2).fit()
model3.summary()

rsq_rnd = smf.ols("RnD~Administration+Marketing+California+Florida+NewYork", data= train_data2).fit().rsquared
ViF_rnd = 1/(1-rsq_rnd)

rsq_adm = smf.ols("Administration~RnD+Marketing+California+Florida+NewYork", data=train_data2).fit().rsquared
ViF_adm = 1/(1-rsq_adm)

rsq_mar = smf.ols("Marketing ~ RnD+Administration+California+Florida+NewYork", data= train_data2).fit().rsquared
ViF_mar = 1/(1-rsq_mar)

sm.graphics.plot_partregress_grid(model2)


model3= smf.ols("Profit~RnD+Marketing+California+Florida+NewYork",data = train_data2).fit()
model3.summary()

finalmodel= smf.ols("Profit~RnD+Marketing+California+Florida+NewYork",data = train_data2).fit()
finalmodel.summary()

train_pred = finalmodel.predict(train_data2)

train_res= train_data2["Profit"]-train_pred

train_rmse = np.sqrt(np.mean(train_res*train_res))

test_pred = finalmodel.predict(test_data)

test_res= test_data["Profit"]- test_pred

test_rmse = np.sqrt(np.mean(test_res*test_res))

startup1= startup.drop(startup.index[[4,24]],axis=0)
bestmodel= smf.ols("Profit~RnD+Marketing+California+Florida+NewYork",data =startup1).fit()
bestmodel.summary()

bestmodel_pred = bestmodel.predict(startup1)

plt.scatter(startup1.Profit,bestmodel_pred,c='r');plt.xlabel("Observed values");plt.ylabel("Fitted values")

plt.scatter(bestmodel_pred,bestmodel.resid_pearson, c='r');plt.axhline(y=0,color='blue');plt.xlabel("Fitted values");plt.ylabel("Residuals")

plt.hist(bestmodel.resid_pearson)

import pylab
import scipy.stats as st

st.probplot(bestmodel.resid_pearson, dist='norm', plot=pylab)

plt.scatter(bestmodel_pred,bestmodel.resid_pearson,c='r');plt.axhline(y=0,color='blue');plt.xlabel("Fitted values");plt.ylabel("Residuals")

