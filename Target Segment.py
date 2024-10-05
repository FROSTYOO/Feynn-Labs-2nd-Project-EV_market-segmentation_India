#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read the excel 
df1 = pd.read_excel(r'C:\Users\sauma\OneDrive\Desktop\EV_Market Analysis India\Dataset\data1.xlsx')

print(df1.head())
# %%
print(df1.info())
print(df1.shape)
print(df1.isnull().sum())
print(df1.describe())

df1.plot.bar(x='State/UT',y='Total Electric Vehicles Registered')
plt.show()

#sort the values 
df1.sort_values(['Total Electric Vehicles Registered'], ascending=False).head(10)
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df2 =pd.read_csv(r'C:\Users\sauma\OneDrive\Desktop\EV_Market Analysis India\Dataset\RS_Session_258_AU_1241_2.i_data_gov_in.csv')
# do the operation df2
print(df2.head())
print(df2.info())
print(df2.shape)
print(df2.isnull().sum())
print(df2.describe())
# plot bar graph for df2 states on y label vs charging stations x label
df2.plot.bar(x='State/UT',y='No. of Electric Vehicle (EV) Chargers Sanctioned')
plt.show()
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df3 =pd.read_excel(r'C:\Users\sauma\OneDrive\Desktop\EV_Market Analysis India\Dataset\evcount.xlsx')
print(df3.head())
print(df3.info())
print(df3.shape)
print(df3.isnull().sum())
print(df3.describe())
#plot a bar graph for df3 states on x label vs electric vehicle count on y label
df3.plot.bar(x='State/UT',y='Total no of Electric Vehicles as on 03.08.2023')
plt.show()
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
df4 =pd.read_csv(r'C:\Users\sauma\OneDrive\Desktop\EV_Market Analysis India\Dataset\demographicData.csv')
print(df4.head())
print(df4.info())
print(df4.shape)
print(df4.isnull().sum())
print(df4.describe())

# Plotting the Car loan status with respect to Marrital Status
sns.countplot(x ='Marrital Status', hue = 'Personal loan',  data = df4, palette = 'Set2')
plt.show()

# %%
import math
a =(df4['Marrital Status'].value_counts()['Married'])/((df4['Marrital Status'].value_counts()['Married'])+(df4['Marrital Status'].value_counts()['Single']))*100
print(math.floor(a),'%')
# %%
df4.rename(columns={'Personal loan':'Car_Loan'},inplace=True)
df4.rename(columns={'Price':'EV_Price'},inplace=True)
labels = ['Car Loan Required','Car Loan not required']
Loan_status = [df4.query('Car_Loan == "Yes"').Car_Loan.count(),df4.query('Car_Loan == "No"').Car_Loan.count()]

explode = [0.1, 0]
palette_color = sns.color_palette('pastel')

plt.pie(Loan_status, labels=labels, colors=palette_color, shadow = "True",
        explode=explode, autopct='%1.1f%%')

plt.show()
# %%
g = sns.catplot(x = "Marrital Status", data=df4, col="Car_Loan", col_wrap=1, aspect=2, kind="count", color='brown')

ax = g.facet_axis(0,0)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2,
            p.get_height() * 1.02,
            format(p.get_height()),
            color='black', rotation='horizontal', size='large')
plt.title('Count of People based on Marital Status buying Car', color='black')
plt.show()
     
# %%
g = sns.catplot(x = "No of Dependents", data=df4, aspect=2, kind="count", color='y')

ax = g.facet_axis(0,0)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2,
            p.get_height() * 1.02,
            format(p.get_height()),
            color='black', rotation='horizontal', size='large')
plt.title('Count of People with Dependants buying Car', color='black')
plt.show()
     

# %%
g = sns.catplot(x = "Profession", data=df4, aspect=2, kind="count", color='cyan')

ax = g.facet_axis(0,0)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2,
            p.get_height() * 1.02,
            format(p.get_height()),
            color='black', rotation='horizontal', size='large')
plt.title('Count of People based on Profession buying Car', color='black')
plt.show()
# %%
SalaryGroup={}
length=df4.shape[0]
SalaryGroup['0-5lakh']=round(df4[(df4['Total Salary']>=100000)&(df4['Total Salary']<500000)].shape[0]*100/length,2)
SalaryGroup['5-15lakh']=round(df4[(df4['Total Salary']>=500000)&(df4['Total Salary']<1500000)].shape[0]*100/length,2)
SalaryGroup['15-25lakh']=round(df4[(df4['Total Salary']>=1500000)&(df4['Total Salary']<2500000)].shape[0]*100/length,2)
SalaryGroup['25-35lakh']=round(df4[(df4['Total Salary']>=2500000)&(df4['Total Salary']<3500000)].shape[0]*100/length,2)
SalaryGroup['35-45lakh']=round(df4[(df4['Total Salary']>=3500000)&(df4['Total Salary']<4500000)].shape[0]*100/length,2)
SalaryGroup['45-55lakh']=round(df4[(df4['Total Salary']>=4500000)&(df4['Total Salary']<5500000)].shape[0]*100/length,2)
fig,ax=plt.subplots()
plots=sns.barplot(x = list(SalaryGroup.keys()), y = list(SalaryGroup.values()))
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 5),
                   textcoords='offset points')
ax.set_ylim(0,50)
plt.title('Percentage of People from each Salary Group Buying Cars in India', color='green')
plt.xlabel('Salary Group', color='r',)
plt.show()
     
# %%
AgeGroup={}
length=df4.shape[0]
AgeGroup['25-30']=round(df4[(df4['Age']>=25)&(df4['Age']<30)].shape[0]*100/length,2)
AgeGroup['30-35']=round(df4[(df4['Age']>=30)&(df4['Age']<35)].shape[0]*100/length,2)
AgeGroup['35-40']=round(df4[(df4['Age']>=35)&(df4['Age']<40)].shape[0]*100/length,2)
AgeGroup['40-45']=round(df4[(df4['Age']>=40)&(df4['Age']<45)].shape[0]*100/length,2)
AgeGroup['45-50']=round(df4[(df4['Age']>=45)&(df4['Age']<50)].shape[0]*100/length,2)
AgeGroup['50-55']=round(df4[(df4['Age']>=50)&(df4['Age']<55)].shape[0]*100/length,2)

fig,ax=plt.subplots()
plots=sns.barplot(x=list(AgeGroup.keys()),y=list(AgeGroup.values()))
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 8),
                   textcoords='offset points')
ax.set_ylim(0,40)
plt.title('Percentage of People from each Age Group Buying Cars in India')
plt.xlabel('Age Group')
plt.show()
# %%
WomenSalaryGroup={}
length=df4.shape[0]
WomenSalaryGroup['Salaried'] = round(df4[(df4['Wife Salary']>0)].shape[0]*100/length,2)
WomenSalaryGroup['Not Salaried'] = round(df4[(df4['Wife Salary'] == 0)].shape[0]*100/length,2)
     
fig,ax=plt.subplots()
plots=sns.barplot(x=list(WomenSalaryGroup.keys()),y=list(WomenSalaryGroup.values()))
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(1, 8),
                   textcoords='offset points')
ax.set_ylim(0,60)
plt.title('Percentage of People Whose Wives are Employed Bought Cars in India')
plt.xlabel('Women Salary Group')
plt.show()
# %%

df4['Wife Working'].value_counts()
df4.drop(df4[df4['Wife Working'] == 'm'].index, inplace=True)
df4.shape
numerical = df4.select_dtypes(include=['float64', 'int64'])
label_encoder = preprocessing.LabelEncoder()
categorical = df4.select_dtypes(include = ["object"])
for cols in categorical:
    # Encode labels in column .
    categorical[cols]= label_encoder.fit_transform(categorical[cols])
    categorical[cols].unique()
     

X = pd.concat([numerical, categorical], axis=1)
X.dtypes
# %%
pca_data = preprocessing.scale(X)
pca = PCA(n_components=12)
pc = pca.fit_transform(X)
names = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11','pc12']
pf = pd.DataFrame(data=pc,columns=names)
pca.explained_variance_ratio_
# %%
loadings = pca.components_
num_pc = pca.n_features_in_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = X.columns.values
loadings_df = loadings_df.set_index('variable')
# %%
plt.rcParams['figure.figsize'] = (10,8)
ax = sns.heatmap(loadings_df, annot=True)
plt.show()
# %%
wcss=[]
for i in range(1,11):
    #preventing random initialization: 'init=k-means++'
    kmeans= KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
     
# %%
plt.figure(figsize=(10,6))
plt.plot(range(1,11),wcss,color='orange', linestyle='solid', marker='o',
          markersize=5)
plt.title('Elbow Method', size=20)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
     
# %%
kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
kmeans.fit(X)
X['cluster'] = kmeans.labels_
print(kmeans.cluster_centers_)
# %%
from collections import Counter
Counter(kmeans.labels_)
Counter({0: 37, 1: 26, 2: 20, 3: 15})
# %%
# Visulazing clusters
sns.scatterplot(data=pf, x="pc1", y="pc2", hue=kmeans.labels_,s = 100)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
          marker="+", c="r", s=200, label="centroids")
plt.legend()
plt.show()
     
# %%
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy as shc
from sklearn.metrics import pairwise_distances
clust2 = AgglomerativeClustering(n_clusters = 4).fit_predict(pairwise_distances(X.transpose()))
plt.figure(figsize = (20,10))
plt.bar(X.columns,clust2)
plt.show()
# %%
df4['Clusters'] = X['cluster']
     

newdf = df4.groupby('Clusters')
     

newdf.get_group(0)
# %%
cluster0 = newdf.get_group(0)
     

cluster0
# %%
cluster0.corr(numeric_only=True)
# %%
plt.figure(figsize=(30,7))
plt.title("Correlation Matrix", fontsize=20)

sns.heatmap(cluster0.corr(numeric_only=True),cmap='RdYlGn',annot=True,vmax=1.0,vmin=-1.0,fmt='g')
#cmap is the color for heatmap, annot=True is to show the correlation matrix value on heatmap
#vmax and vmin are the threshold for the heatmap and fmt will create a scale
plt.show()
# %%
plt.figure(figsize=(3, 3))
hist = cluster0['Age'].hist(bins=3)
plt.show()
# %%
plt.figure(figsize=(3, 3))
hist = cluster0['No of Dependents'].hist(bins=2)
plt.show()
     
# %%
plt.figure(figsize=(8, 5))
hist = cluster0['Total Salary'].hist(bins=10)
plt.show()
     
# %%
