#-- GEO1001.2020--hw01
#-- [Theodoros Papakostas] 
#-- [5287928]


import xlrd
import statistics 
import numpy as np 
import pandas as pd 
from pandas import DataFrame
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as stats
import xlsxwriter
import scipy.stats
#import scipy as stats
#read data, in dataframes
path1='HEAT-A_final.csv'
path2='HEAT-B_final.csv'
path3='HEAT-C_final.csv'
path4='HEAT-D_final.csv'
path5='HEAT-E_final.csv'
df1=pd.read_csv(path1, skiprows =5, sep=";",encoding="UTF-8",names=[x for x in range(19)])
df2=pd.read_csv(path2, skiprows =5, sep=";",encoding="UTF-8",names=[x for x in range(19)])
df3=pd.read_csv(path3, skiprows =5, sep=";",encoding="UTF-8",names=[x for x in range(19)])
df4=pd.read_csv(path4, skiprows =5, sep=";",encoding="UTF-8",names=[x for x in range(19)])
df5=pd.read_csv(path5, skiprows =5, sep=";",encoding="UTF-8",names=[x for x in range(19)])
#Stats for dataset HEATA
a1 = df1.mean()
b1 = df1.var()
c1 = df1.std()
a1 = a1.to_numpy()
b1 = b1.to_numpy()
c1 = c1.to_numpy()
#print(a1,b1,c1)
#Stats for dataset HEATB
a2 = df2.mean()
b2 = df2.var()
c2 = df2.std()
a2 = a2.to_numpy()
b2 = b2.to_numpy()
c2 = c2.to_numpy()
#print(a2,b2,c2)
#Stats for dataset HEATC
a3 = df3.mean()
b3 = df3.var()
c3 = df3.std()
a3 = a3.to_numpy()
b3 = b3.to_numpy()
c3 = c3.to_numpy()
#print(a3,b3,c3)
#Stats for dataset HEATD 
a4 = df4.mean()
b4 = df4.var()
c4 = df4.std()
a4 = a4.to_numpy()
b4 = b4.to_numpy()
c4 = c4.to_numpy()
#print(a4,b4,c4)
#Stats for dataset HEATE
a5 = df5.mean()
b5 = df5.var()
c5 = df5.std()
a5 = a5.to_numpy()
b5 = b5.to_numpy()
c5 = c5.to_numpy()
#print(a5,b5,c5)
np.savetxt("Stats_5_sensors.csv", [a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5], delimiter=",")
#Temperature variable of the datasets,histogram
temp1=df1[4]
temp2=df2[4]
temp3=df3[4]
temp4=df4[4]
temp5=df5[4]
fig1=plt.figure(1)
plt.hist([temp1,temp2,temp3,temp4,temp5],bins=5,label=['HEAT A','HEAT B','HEAT C','HEAT D','HEAT E'])
plt.xlabel('Temperature')
plt.ylabel('n0 of timeperiods')
plt.title('Histogram - 5 bins')
plt.legend()
fig2=plt.figure(2)
plt.hist([temp1,temp2,temp3,temp4,temp5],bins=50,label=['HEAT A','HEAT B','HEAT C','HEAT D','HEAT E'])
plt.xlabel('Temperature')
plt.ylabel('n0 of timeperiods')
plt.title('Histogram - 50 bins')
plt.legend()
#frequency polygons
fig3=plt.figure(3)
y1,edges = np.histogram(temp1,bins=27)
y2,edges = np.histogram(temp2,bins=27)
y3,edges = np.histogram(temp3,bins=27)
y4,edges = np.histogram(temp4,bins=27)
y5,edges = np.histogram(temp5,bins=27)
centers = 0.5*(edges[1:]+ edges[:-1])
plt.plot(centers,y1,'-*',label='HEAT A')
plt.plot(centers,y2,'-*',label='HEAT B')
plt.plot(centers,y3,'-*',label='HEAT C')
plt.plot(centers,y4,'-*',label='HEAT D')
plt.plot(centers,y5,'-*',label='HEAT E')
plt.xlabel('Temperature')
plt.ylabel('n0 of timeperiods')
plt.title('Frequency Polygons')
plt.legend()
#boxplots for WindDirection,WindSpeed,Temperature of the 5 sensors
wd1=df1[0]
wd2=df2[0]
wd3=df3[0]
wd4=df4[0]
wd5=df5[0]
ws1=df1[1]
ws2=df2[1]
ws3=df3[1]
ws4=df4[1]
ws5=df5[1]
fig4=plt.figure(4)
plt.boxplot([wd1,wd2,wd3,wd4,wd5],showmeans=True)
plt.ylabel('Wind Direction')
plt.title('Boxplots(Wind Direction) of the 5 sensors')
fig5=plt.figure(5)
plt.boxplot([ws1,ws2,ws3,ws4,ws5],showmeans=True)
plt.ylabel('Wind Speed')
plt.title('Boxplots(Wind Speed) of the 5 sensors')
fig6=plt.figure(6)
plt.boxplot([temp1,temp2,temp3,temp4,temp5],showmeans=True)
plt.ylabel('Temperature')
plt.title('Boxplots(Temperature) of the 5 sensors')
#PMF, 5 sensors Temperature
def pmf(sample):
	c = sample.value_counts()
	p = c/len(sample)
	return p
dff1 = pmf(temp1)
c1 = dff1.sort_index()
dff2 = pmf(temp2)
c2 = dff2.sort_index()
dff3 = pmf(temp3)
c3 = dff3.sort_index()
dff4 = pmf(temp4)
c4 = dff4.sort_index()
dff5 = pmf(temp5)
c5 = dff5.sort_index()
fig, axs=plt.subplots(5)
fig.suptitle('PMF, 5 sensors Temperature')
axs[0].bar(c1.index,c1)
axs[1].bar(c2.index,c2)
axs[2].bar(c3.index,c3)
axs[3].bar(c4.index,c4)
axs[4].bar(c5.index,c5)
#PDF, 5 sensors Temperature
fig, axs = plt.subplots(5)
fig.suptitle('PDF, 5 sensors Temperature')
axs[0].hist([temp1.astype(float)], density=True, alpha=0.7, rwidth=0.85,bins=27)
sns.distplot([temp1.astype(float)], ax=axs[0],bins=27)
axs[1].hist([temp2.astype(float)], density=True, alpha=0.7, rwidth=0.85,bins=27)
sns.distplot([temp2.astype(float)], ax=axs[1],bins=27)
axs[2].hist([temp3.astype(float)], density=True, alpha=0.7, rwidth=0.85,bins=27)
sns.distplot([temp3.astype(float)], ax=axs[2],bins=27)
axs[3].hist([temp4.astype(float)], density=True, alpha=0.7, rwidth=0.85,bins=27)
sns.distplot([temp4.astype(float)], ax=axs[3],bins=27)
axs[4].hist([temp5.astype(float)], density=True, alpha=0.7, rwidth=0.85,bins=27)
sns.distplot([temp5.astype(float)], ax=axs[4],bins=27)
#CDF, 5 sensors Temperature
fig, axs = plt.subplots(5)
fig.suptitle('CDF, 5 sensors Temperature')
a1=axs[0].hist([temp1.astype(float)],cumulative=True,alpha=0.7, rwidth=0.85,bins=27)
axs[0].plot(a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2,a1[0], color='k')
a2=axs[1].hist([temp2.astype(float)],cumulative=True,alpha=0.7, rwidth=0.85,bins=27)
axs[1].plot(a2[1][1:]-(a2[1][1:]-a2[1][:-1])/2,a2[0], color='k')
a3=axs[2].hist([temp3.astype(float)],cumulative=True,alpha=0.7, rwidth=0.85,bins=27)
axs[2].plot(a3[1][1:]-(a3[1][1:]-a3[1][:-1])/2,a3[0], color='k')
a4=axs[3].hist([temp4.astype(float)],cumulative=True,alpha=0.7, rwidth=0.85,bins=27)
axs[3].plot(a4[1][1:]-(a4[1][1:]-a4[1][:-1])/2,a4[0], color='k')
a5=axs[4].hist([temp5.astype(float)],cumulative=True,alpha=0.7, rwidth=0.85,bins=27)
axs[4].plot(a5[1][1:]-(a5[1][1:]-a5[1][:-1])/2,a5[0], color='k')
#PDF, 5 sensors Wind Speed, with Kernel Density Estimation
fig, axs = plt.subplots(2,sharex=True,sharey=True)
axs[0].title.set_text('PDF,Wind speed sensor A')
axs[1].title.set_text('Kernel Density Estimation, Wind speed sensor A')
axs[0].hist([ws1.astype(float)],density=True,alpha=0.7, rwidth=0.85)
sns.distplot([ws1.astype(float)], ax=axs[0])
kernel = stats.gaussian_kde(ws1)
x_eval = np.linspace(-10, 10, num=200)
axs[1].plot(x_eval,kernel(x_eval))
fig, axs = plt.subplots(2,sharex=True,sharey=True)
axs[0].title.set_text('PDF,Wind speed sensor B')
axs[1].title.set_text('Kernel Density Estimation, Wind speed sensor B')
axs[0].hist([ws2.astype(float)],density=True,alpha=0.7, rwidth=0.85)
sns.distplot([ws2.astype(float)], ax=axs[0])
kernel = stats.gaussian_kde(ws2)
x_eval = np.linspace(-10, 10, num=200)
axs[1].plot(x_eval,kernel(x_eval))
fig, axs = plt.subplots(2,sharex=True,sharey=True)
axs[0].title.set_text('PDF,Wind speed sensor C')
axs[1].title.set_text('Kernel Density Estimation, Wind speed sensor C')
axs[0].hist([ws3.astype(float)],density=True,alpha=0.7, rwidth=0.85)
sns.distplot([ws3.astype(float)], ax=axs[0])
kernel = stats.gaussian_kde(ws3)
x_eval = np.linspace(-10, 10, num=200)
axs[1].plot(x_eval,kernel(x_eval))
fig, axs = plt.subplots(2,sharex=True,sharey=True)
axs[0].title.set_text('PDF,Wind speed sensor D')
axs[1].title.set_text('Kernel Density Estimation, Wind speed sensor D')
axs[0].hist([ws4.astype(float)],density=True,alpha=0.7, rwidth=0.85)
sns.distplot([ws4.astype(float)], ax=axs[0])
kernel = stats.gaussian_kde(ws4)
x_eval = np.linspace(-10, 10, num=200)
axs[1].plot(x_eval,kernel(x_eval))
fig, axs = plt.subplots(2,sharex=True,sharey=True)
axs[0].title.set_text('PDF,Wind speed sensor E')
axs[1].title.set_text('Kernel Density Estimation, Wind speed sensor E')
axs[0].hist([ws5.astype(float)],density=True,alpha=0.7, rwidth=0.85)
sns.distplot([ws5.astype(float)], ax=axs[0])
kernel = stats.gaussian_kde(ws5)
x_eval = np.linspace(-10, 10, num=200)
axs[1].plot(x_eval,kernel(x_eval))
#Correlations, 5 sensors Temperature, Wet Bulb Globe & Crosswind Speed
Cws1=df1[2]
Cws2=df2[2]
Cws3=df3[2]
Cws4=df4[2]
Cws5=df5[2]
WBG1=df1[16]
WBG2=df2[16]
WBG3=df3[16]
WBG4=df4[16]
WBG5=df5[16]
#Interpolation of sensors' data
#A-B
temp1i = np.interp(np.linspace(0,len(temp2),len(temp2)),np.linspace(0,len(temp1),len(temp1)),temp1)
WBG1 = np.interp(np.linspace(0,len(WBG2),len(WBG2)),np.linspace(0,len(WBG1),len(WBG1)),WBG1)
Cws1 = np.interp(np.linspace(0,len(Cws2),len(Cws2)),np.linspace(0,len(Cws1),len(Cws1)),Cws1)
#Coef A-B
pcoef_temp12 = stats.pearsonr(temp1i,temp2)
prcoef_temp12= stats.spearmanr(temp1i,temp2) 
pcoef_WBG12 = stats.pearsonr(WBG1,WBG2)
prcoef_WBG12= stats.spearmanr(WBG1,WBG2) 
pcoef_Cws12 = stats.pearsonr(Cws1,Cws2)
prcoef_Cws12= stats.spearmanr(Cws1,Cws2)
print(pcoef_temp12,prcoef_temp12)
print(np.cov(temp1i,temp2))
print(pcoef_WBG12,prcoef_WBG12)
print(np.cov(WBG1,WBG2))
print(pcoef_Cws12,prcoef_Cws12)
print(np.cov(Cws1,Cws2))
#A-C
temp1i = np.interp(np.linspace(0,len(temp3),len(temp3)),np.linspace(0,len(temp1),len(temp1)),temp1)
WBG1 = np.interp(np.linspace(0,len(WBG3),len(WBG3)),np.linspace(0,len(WBG1),len(WBG1)),WBG1)
Cws1 = np.interp(np.linspace(0,len(Cws3),len(Cws3)),np.linspace(0,len(Cws1),len(Cws1)),Cws1)
#Coef A-C
pcoef_temp13 = stats.pearsonr(temp1i,temp3)
prcoef_temp13= stats.spearmanr(temp1i,temp3) 
pcoef_WBG13 = stats.pearsonr(WBG1,WBG3)
prcoef_WBG13= stats.spearmanr(WBG1,WBG3) 
pcoef_Cws13 = stats.pearsonr(Cws1,Cws3)
prcoef_Cws13= stats.spearmanr(Cws1,Cws3)
print(pcoef_temp13,prcoef_temp13)
print(np.cov(temp1i,temp3))
print(pcoef_WBG13,prcoef_WBG13)
print(np.cov(WBG1,WBG3))
print(pcoef_Cws13,prcoef_Cws13)
print(np.cov(Cws1,Cws3))
#A-D
temp1i = np.interp(np.linspace(0,len(temp4),len(temp4)),np.linspace(0,len(temp1),len(temp1)),temp1)
WBG1 = np.interp(np.linspace(0,len(WBG4),len(WBG4)),np.linspace(0,len(WBG1),len(WBG1)),WBG1)
Cws1 = np.interp(np.linspace(0,len(Cws4),len(Cws4)),np.linspace(0,len(Cws1),len(Cws1)),Cws1)
#Coef A-D
pcoef_temp14 = stats.pearsonr(temp1i,temp4)
prcoef_temp14= stats.spearmanr(temp1i,temp4) 
pcoef_WBG14 = stats.pearsonr(WBG1,WBG4)
prcoef_WBG14= stats.spearmanr(WBG1,WBG4) 
pcoef_Cws14 = stats.pearsonr(Cws1,Cws4)
prcoef_Cws14= stats.spearmanr(Cws1,Cws4)
print(pcoef_temp14,prcoef_temp14)
print(np.cov(temp1i,temp4))
print(pcoef_WBG14,prcoef_WBG14)
print(np.cov(WBG1,WBG4))
print(pcoef_Cws14,prcoef_Cws14)
print(np.cov(Cws1,Cws4))
#A-E
temp1i = np.interp(np.linspace(0,len(temp5),len(temp5)),np.linspace(0,len(temp1),len(temp1)),temp1)
WBG1 = np.interp(np.linspace(0,len(WBG5),len(WBG5)),np.linspace(0,len(WBG1),len(WBG1)),WBG1)
Cws1 = np.interp(np.linspace(0,len(Cws5),len(Cws5)),np.linspace(0,len(Cws1),len(Cws1)),Cws1)
#Coef A-E
pcoef_temp15 = stats.pearsonr(temp1i,temp5)
prcoef_temp15= stats.spearmanr(temp1i,temp5) 
pcoef_WBG15 = stats.pearsonr(WBG1,WBG5)
prcoef_WBG15= stats.spearmanr(WBG1,WBG5) 
pcoef_Cws15 = stats.pearsonr(Cws1,Cws5)
prcoef_Cws15= stats.spearmanr(Cws1,Cws5)
print(pcoef_temp15,prcoef_temp15)
print(np.cov(temp1i,temp5))
print(pcoef_WBG15,prcoef_WBG15)
print(np.cov(WBG1,WBG5))
print(pcoef_Cws15,prcoef_Cws15)
print(np.cov(Cws1,Cws5))
# B-C
temp2i = np.interp(np.linspace(0,len(temp3),len(temp3)),np.linspace(0,len(temp3),len(temp2)),temp2)
WBG2 = np.interp(np.linspace(0,len(WBG3),len(WBG3)),np.linspace(0,len(WBG3),len(WBG2)),WBG2)
Cws2 = np.interp(np.linspace(0,len(Cws3),len(Cws3)),np.linspace(0,len(Cws3),len(Cws2)),Cws2)
#Coef B-C
pcoef_temp23 = stats.pearsonr(temp2i,temp3)
prcoef_temp23= stats.spearmanr(temp2i,temp3) 
pcoef_WBG23 = stats.pearsonr(WBG2,WBG3)
prcoef_WBG23= stats.spearmanr(WBG2,WBG3) 
pcoef_Cws23 = stats.pearsonr(Cws2,Cws3)
prcoef_Cws23= stats.spearmanr(Cws2,Cws3)
print(pcoef_temp23,prcoef_temp23)
print(np.cov(temp2i,temp3))
print(pcoef_WBG23,prcoef_WBG23)
print(np.cov(WBG2,WBG3))
print(pcoef_Cws23,prcoef_Cws23)
print(np.cov(Cws2,Cws3))
# B-D
temp2i = np.interp(np.linspace(0,len(temp4),len(temp4)),np.linspace(0,len(temp2),len(temp2)),temp2)
WBG2 = np.interp(np.linspace(0,len(WBG4),len(WBG4)),np.linspace(0,len(WBG2),len(WBG2)),WBG2)
Cws2 = np.interp(np.linspace(0,len(Cws4),len(Cws4)),np.linspace(0,len(Cws2),len(Cws2)),Cws2)
#Coef B-D
pcoef_temp24 = stats.pearsonr(temp2i,temp4)
prcoef_temp24= stats.spearmanr(temp2i,temp4) 
pcoef_WBG24 = stats.pearsonr(WBG2,WBG4)
prcoef_WBG24= stats.spearmanr(WBG2,WBG4) 
pcoef_Cws24 = stats.pearsonr(Cws2,Cws4)
prcoef_Cws24= stats.spearmanr(Cws2,Cws4)
print(pcoef_temp24,prcoef_temp24)
print(np.cov(temp2i,temp4))
print(pcoef_WBG24,prcoef_WBG24)
print(np.cov(WBG2,WBG4))
print(pcoef_Cws24,prcoef_Cws24)
print(np.cov(Cws2,Cws4))
# B-E
temp2i = np.interp(np.linspace(0,len(temp5),len(temp5)),np.linspace(0,len(temp2),len(temp2)),temp2)
WBG2 = np.interp(np.linspace(0,len(WBG5),len(WBG5)),np.linspace(0,len(WBG2),len(WBG2)),WBG2)
Cws2 = np.interp(np.linspace(0,len(Cws5),len(Cws5)),np.linspace(0,len(Cws2),len(Cws2)),Cws2)
#Coef B-E
pcoef_temp25 = stats.pearsonr(temp2i,temp5)
prcoef_temp25= stats.spearmanr(temp2i,temp5) 
pcoef_WBG25 = stats.pearsonr(WBG2,WBG5)
prcoef_WBG25= stats.spearmanr(WBG2,WBG5) 
pcoef_Cws25 = stats.pearsonr(Cws2,Cws5)
prcoef_Cws25= stats.spearmanr(Cws2,Cws5)
print(pcoef_temp25,prcoef_temp25)
print(np.cov(temp2i,temp5))
print(pcoef_WBG25,prcoef_WBG25)
print(np.cov(WBG2,WBG5))
print(pcoef_Cws25,prcoef_Cws25)
print(np.cov(Cws2,Cws5))
# C-D
temp3i = np.interp(np.linspace(0,len(temp4),len(temp4)),np.linspace(0,len(temp3),len(temp3)),temp3)
WBG3 = np.interp(np.linspace(0,len(WBG4),len(WBG4)),np.linspace(0,len(WBG3),len(WBG3)),WBG3)
Cws3 = np.interp(np.linspace(0,len(Cws4),len(Cws4)),np.linspace(0,len(Cws3),len(Cws3)),Cws3)
#Coef C-D
pcoef_temp34 = stats.pearsonr(temp3i,temp4)
prcoef_temp34= stats.spearmanr(temp3i,temp4) 
pcoef_WBG34 = stats.pearsonr(WBG3,WBG4)
prcoef_WBG34= stats.spearmanr(WBG3,WBG4) 
pcoef_Cws34 = stats.pearsonr(Cws3,Cws4)
prcoef_Cws34= stats.spearmanr(Cws3,Cws4)
print(pcoef_temp34,prcoef_temp34)
print(np.cov(temp3i,temp4))
print(pcoef_WBG34,prcoef_WBG34)
print(np.cov(WBG3,WBG4))
print(pcoef_Cws34,prcoef_Cws34)
print(np.cov(Cws3,Cws4))
# C-E
temp3i = np.interp(np.linspace(0,len(temp5),len(temp5)),np.linspace(0,len(temp3),len(temp3)),temp3)
WBG3 = np.interp(np.linspace(0,len(WBG5),len(WBG5)),np.linspace(0,len(WBG3),len(WBG3)),WBG3)
Cws3 = np.interp(np.linspace(0,len(Cws5),len(Cws5)),np.linspace(0,len(Cws3),len(Cws3)),Cws3)
#Coef C-E
pcoef_temp35 = stats.pearsonr(temp3i,temp5)
prcoef_temp35= stats.spearmanr(temp3i,temp5) 
pcoef_WBG35 = stats.pearsonr(WBG3,WBG5)
prcoef_WBG35= stats.spearmanr(WBG3,WBG5) 
pcoef_Cws35 = stats.pearsonr(Cws3,Cws5)
prcoef_Cws35= stats.spearmanr(Cws3,Cws5)
print(pcoef_temp35,prcoef_temp35)
print(np.cov(temp3i,temp5))
print(pcoef_WBG35,prcoef_WBG35)
print(np.cov(WBG3,WBG5))
print(pcoef_Cws35,prcoef_Cws35)
print(np.cov(Cws3,Cws5))
# D-E
temp4i = np.interp(np.linspace(0,len(temp5),len(temp5)),np.linspace(0,len(temp4),len(temp4)),temp4)
WBG4 = np.interp(np.linspace(0,len(WBG5),len(WBG5)),np.linspace(0,len(WBG4),len(WBG4)),WBG4)
Cws4 = np.interp(np.linspace(0,len(Cws5),len(Cws5)),np.linspace(0,len(Cws4),len(Cws4)),Cws4)
#Coef D-E
pcoef_temp45 = stats.pearsonr(temp4i,temp5)
prcoef_temp45= stats.spearmanr(temp4i,temp5) 
pcoef_WBG45 = stats.pearsonr(WBG4,WBG5)
prcoef_WBG45= stats.spearmanr(WBG4,WBG5) 
pcoef_Cws45 = stats.pearsonr(Cws4,Cws5)
prcoef_Cws45= stats.spearmanr(Cws4,Cws5)
print(pcoef_temp45,prcoef_temp45)
print(np.cov(temp4i,temp5))
print(pcoef_WBG45,prcoef_WBG45)
print(np.cov(WBG4,WBG5))
print(pcoef_Cws45,prcoef_Cws45)
print(np.cov(Cws4,Cws5))
#Coeffs scatter plots
PearsonTemp=[pcoef_temp12[0],pcoef_temp13[0],pcoef_temp14[0],pcoef_temp15[0],pcoef_temp23[0],pcoef_temp24[0],pcoef_temp25[0],pcoef_temp34[0],pcoef_temp35[0],pcoef_temp45[0]]
SpearmanTemp=[prcoef_temp12[0],prcoef_temp13[0],prcoef_temp14[0],prcoef_temp15[0],prcoef_temp23[0],prcoef_temp24[0],prcoef_temp25[0],prcoef_temp34[0],prcoef_temp35[0],prcoef_temp45[0]]
A1=[1,2,3,4,5,6,7,8,9,10]
labels=["AB","AC","AD","AE","BC","BD","BE","CD","CE","DE"]
fig = plt.figure()
plt.title("Pearson & Spearman coefficients, Temperatures")
plt.xticks(A1,labels)
a= plt.scatter(A1,PearsonTemp)
b= plt.scatter(A1,SpearmanTemp)
plt.legend((a,b),("Pearson","Spearman"))
PearsonWBG=[pcoef_WBG12[0],pcoef_WBG13[0],pcoef_WBG14[0],pcoef_WBG15[0],pcoef_WBG23[0],pcoef_WBG24[0],pcoef_WBG25[0],pcoef_WBG34[0],pcoef_WBG35[0],pcoef_WBG45[0]]
SpearmanWBG=[prcoef_WBG12[0],prcoef_WBG13[0],prcoef_WBG14[0],prcoef_WBG15[0],prcoef_WBG23[0],prcoef_WBG24[0],prcoef_WBG25[0],prcoef_WBG34[0],prcoef_WBG35[0],prcoef_WBG45[0]]
A1=[1,2,3,4,5,6,7,8,9,10]
labels=["AB","AC","AD","AE","BC","BD","BE","CD","CE","DE"]
fig = plt.figure()
plt.title("Pearson & Spearman coefficients, Wet Bulb Globe Temperature")
plt.xticks(A1,labels)
a =plt.scatter(A1,PearsonWBG)
b =plt.scatter(A1,SpearmanWBG)
plt.legend((a,b),("Pearson","Spearman"))
PearsonCws=[pcoef_Cws12[0],pcoef_Cws13[0],pcoef_Cws14[0],pcoef_Cws15[0],pcoef_Cws23[0],pcoef_Cws24[0],pcoef_Cws25[0],pcoef_Cws34[0],pcoef_Cws35[0],pcoef_Cws45[0]]
SpearmanCws=[prcoef_Cws12[0],prcoef_Cws13[0],prcoef_Cws14[0],prcoef_Cws15[0],prcoef_Cws23[0],prcoef_Cws24[0],prcoef_Cws25[0],prcoef_Cws34[0],prcoef_Cws35[0],prcoef_Cws45[0]]
A1=[1,2,3,4,5,6,7,8,9,10]
labels=["AB","AC","AD","AE","BC","BD","BE","CD","CE","DE"]
fig = plt.figure()
plt.title("Pearson & Spearman coefficients, CrossWind Speed")
plt.xticks(A1,labels)
a =plt.scatter(A1,PearsonCws)
b =plt.scatter(A1,SpearmanCws)
plt.legend((a,b),("Pearson","Spearman"))
#****************
#CDF, Wind Speed of the 5 sensors
fig, axs = plt.subplots(5)
fig.suptitle('CDF, 5 sensors Wind Speed')
a1=axs[0].hist([ws1.astype(float)],cumulative=True,alpha=0.7, rwidth=0.85,bins=27)
axs[0].plot(a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2,a1[0], color='k')
a2=axs[1].hist([ws2.astype(float)],cumulative=True,alpha=0.7, rwidth=0.85,bins=27)
axs[1].plot(a2[1][1:]-(a2[1][1:]-a2[1][:-1])/2,a2[0], color='k')
a3=axs[2].hist([ws3.astype(float)],cumulative=True,alpha=0.7, rwidth=0.85,bins=27)
axs[2].plot(a3[1][1:]-(a3[1][1:]-a3[1][:-1])/2,a3[0], color='k')
a4=axs[3].hist([ws4.astype(float)],cumulative=True,alpha=0.7, rwidth=0.85,bins=27)
axs[3].plot(a4[1][1:]-(a4[1][1:]-a4[1][:-1])/2,a4[0], color='k')
a5=axs[4].hist([ws5.astype(float)],cumulative=True,alpha=0.7, rwidth=0.85,bins=27)
axs[4].plot(a5[1][1:]-(a5[1][1:]-a5[1][:-1])/2,a5[0], color='k')
#********************************
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h
Cit1=(mean_confidence_interval(temp1, confidence=0.95))
Cit2=(mean_confidence_interval(temp2, confidence=0.95))
Cit3=(mean_confidence_interval(temp3, confidence=0.95))
Cit4=(mean_confidence_interval(temp4, confidence=0.95))
Cit5=(mean_confidence_interval(temp5, confidence=0.95))
Ciws1=(mean_confidence_interval(ws1, confidence=0.95))
Ciws2=(mean_confidence_interval(ws2, confidence=0.95))
Ciws3=(mean_confidence_interval(ws3, confidence=0.95))
Ciws4=(mean_confidence_interval(ws4, confidence=0.95))
Ciws5=(mean_confidence_interval(ws5, confidence=0.95))
Workbook=xlsxwriter.Workbook("Confidence intervals.xlsx")
Worksheet=Workbook.add_worksheet()
Worksheet.write("B1","Confidence Intervals,Temperature")
Worksheet.write("C1","Confidence Intervals,Wind Speed")
Worksheet.write("B2",Cit1[0])
Worksheet.write("B3",Cit1[1])
Worksheet.write("B4",Cit1[2])
Worksheet.write("C2",Ciws1[0])
Worksheet.write("C3",Ciws1[1])
Worksheet.write("C4",Ciws1[2])
Worksheet.write("B5",Cit2[0])
Worksheet.write("B6",Cit2[1])
Worksheet.write("B7",Cit2[2])
Worksheet.write("C5",Ciws2[0])
Worksheet.write("C6",Ciws2[1])
Worksheet.write("C7",Ciws2[2])
Worksheet.write("B8",Cit3[0])
Worksheet.write("B9",Cit3[1])
Worksheet.write("B10",Cit3[2])
Worksheet.write("C8",Ciws3[0])
Worksheet.write("C9",Ciws3[1])
Worksheet.write("C10",Ciws3[2])
Worksheet.write("B11",Cit4[0])
Worksheet.write("B12",Cit4[1])
Worksheet.write("B13",Cit4[2])
Worksheet.write("C11",Ciws4[0])
Worksheet.write("C12",Ciws4[1])
Worksheet.write("C13",Ciws4[2])
Worksheet.write("B14",Cit5[0])
Worksheet.write("B15",Cit5[1])
Worksheet.write("B16",Cit5[2])
Worksheet.write("C14",Ciws5[0])
Worksheet.write("C15",Ciws5[1])
Worksheet.write("C16",Ciws5[2])
Workbook.close()
t1,p1 = stats.ttest_ind(temp5,temp4)
#print(t1,p1)
t2,p2 = stats.ttest_ind(ws5,ws4)
#print(t2,p2)
t3,p3 = stats.ttest_ind(temp4,temp3)
#print(t3,p3)
t4,p4 = stats.ttest_ind(ws4,ws3)
#print(t4,p4)
t5,p5 = stats.ttest_ind(temp3,temp2)
#print(t5,p5)
t6,p6 = stats.ttest_ind(ws3,ws2)
#print(t5,p5)
t7,p7 = stats.ttest_ind(temp2,temp1)
#print(t7,p7)
t8,p8 = stats.ttest_ind(ws2,ws1)
#print(t8,p8)
np.savetxt("Student_test.csv", [[t1,p1],[t2,p2],[t3,p3],[t4,p4],[t5,p5],[t6,p6],[t7,p7],[t8,p8]], delimiter=",")
#****************BONUS*****************
def average_temperature(data):
    temperature=[data[0:72].mean(),data[72:144].mean(),data[144:216].mean(),data[216:288].mean(),data[288:360].mean(),data[360:432].mean(),data[432:504].mean(),
    data[504:576].mean(),data[576:648].mean(),data[648:720].mean(),data[720:792].mean(),data[792:864].mean(),data[864:936].mean(),data[936:1008].mean(),
    data[1008:1080].mean(),data[1080:1152].mean(),data[1152:1224].mean(),data[1224:1296].mean(),data[1296:1368].mean(),data[1368:1440].mean(),data[1440:1512].mean()
    ,data[1512:1584].mean(),data[1584:1656].mean(),data[1656:1728].mean(),data[1728:1800].mean(),data[1800:1872].mean(),data[1872:1944].mean(),data[1944:2016].mean(),
    data[2016:2088].mean(),data[2088:2160].mean(),data[2160:2232].mean(),data[2232:2304].mean(),data[2304:2376].mean(),data[2376:2448].mean(),data[2448:2476].mean()]
    dates=["june 10","june 11","june 12","june 13","june 14","june 15","june 16","june 17","june 18","june 19","june 20",
    "june 21","june 22","june 23","june 24","june 25","june 26","june 27","june 28","june 29","june 30","july 1","july 2",
    "july 3","july 4","july 5","july 6","july 7","july 8","july 9","july 10","july 11","july 12","july 13","july 14"]    
    d={'Temperature':temperature,'Date':dates}
    df_print=pd.DataFrame(d)
    df_print = df_print.sort_values(by =['Temperature', 'Date'],ascending=False)
    return (df_print)
av1 = (average_temperature(df1[4]))
av1 = av1.to_numpy()
print (av1[0])
print (av1[34])
av2 = (average_temperature(df2[4]))
av2 = av2.to_numpy()
print (av2[0])
print (av2[34])
av3 = (average_temperature(df3[4]))
av3 = av3.to_numpy()
print (av3[0])
print (av3[34])
av4 = (average_temperature(df4[4]))
av4 = av4.to_numpy()
print (av4[0])
print (av4[34])
av5 = (average_temperature(df5[4]))
av5 = av5.to_numpy()
print (av5[0])
print (av5[34])

plt.show()
