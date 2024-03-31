import numpy as np
from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing

# import openpyxl

titanic = pd.read_csv('D:/DMProject/titanic_data.csv')
print(titanic)
print('------')

# 1.1 数据描述
titanic_desc = titanic.describe()
print(titanic.head())
print(titanic_desc)
print(titanic.info())
print('------')

# titanic_df = pd.DataFrame(titanic_desc)
# titanic_df.to_excel("D:\\DMProject\\Excel\\titanic_desc.xlsx", encoding='utf-8', index=False)
# Excel导出(titanic_desc)

# 1.2 数据预处理 (缺失值处理 异常值处理 特征选取)
missing = titanic.isnull().sum()
print('Missing:')
print(missing)
titanic = titanic.drop(columns='Cabin')
print('\n')
print('Pclass:')
print(titanic[titanic['Fare'].isnull()]['Pclass'])
titanic['Fare'].fillna(titanic[titanic['Pclass'] == 3]['Fare'].mean(), inplace=True)
print('\n')
print('Mode:')
print(titanic['Embarked'].value_counts())
titanic['Embarked'].fillna('S', inplace=True)
titanic = titanic.drop(columns=['PassengerId', 'Ticket'])
print(titanic.info())
print('------')

for i in range(0, 2):
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
    titanic.iloc[:, i].plot.box(vert=False, grid=True, color=color, ax=ax1, label='sample data')
    plt.show()

for i in range(4, 8):
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
    titanic.iloc[:, i].plot.box(vert=False, grid=True, color=color, ax=ax1, label='sample data')
    plt.show()

neg = list(titanic.columns.values)
del neg[2:4]
del neg[-1]
print(neg)

for i in neg:
    q1 = titanic_desc[i].iloc[4]
    q3 = titanic_desc[i].iloc[6]
    EI = q3 - q1
    mi = q1 - 1.5 * EI
    ma = q3 + 1.5 * EI
    # 计算分位差

    error = titanic[i][np.array((titanic[i] < mi) | (titanic[i] > ma))]
    print('异常值共%i条' % len(error))
    print('------')

# print('------')

# 2. 年龄与票价的关系
plt.scatter(titanic['Age'], titanic['Fare'], c='#D6D1C4', s=24)
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Relationship between Age and Fare')
plt.show()

# 3. 票价与舱位的关系
plt.scatter(titanic['Pclass'], titanic['Fare'], c='#D6D1C4', s=24)
plt.xlabel('Pclass')
plt.ylabel('Fare')
plt.title('Relationship between Pclass and Fare')
plt.show()

# 4. 生还几率
# 4.1 乘客生还几率
count = titanic['Survived'].value_counts()
print(count)
# 总体生还情况

# 4.2 总体生还几率
plt.figure(figsize=(6, 5))
plt.pie(count, autopct='%.2f%%', labels=['Death', 'Survivable'],
        shadow=True, explode=[0, 0.1], textprops=dict(size=13), colors=['#F1D9CF', '#DBDFE2'])
plt.title('Overall Survival Rate')
plt.show()

# 4.3 不同性别乘客生还几率
sex_count = titanic.groupby(by='Sex')['Survived'].value_counts()
print(sex_count)
# 不同性别乘客的生还情况

n_sex = titanic['Sex'].value_counts()
print(n_sex)
plt.pie(n_sex, autopct='%.2f%%', labels=['Male', 'Female'], shadow=True, explode=[0, 0.1],
        textprops=dict(size=13), colors=['#F1D9CF', '#DBDFE2'])
plt.title('Male-Female Ratio')
plt.show()
# 乘客性别比例

plt.figure(figsize=(2 * 5, 5))
axes1 = plt.subplot(1, 2, 1)
axes1.pie(sex_count.loc['female'][::-1], autopct='%.2f%%', labels=['Death', 'Survival'],
          shadow=True, explode=[0, 0.1], textprops=dict(size=13), colors=['#E79482', '#F6CCB3'], startangle=90)
axes1.set_title('Female Survival Rate')
axes2 = plt.subplot(1, 2, 2)
axes2.pie(sex_count.loc['male'], autopct='%.2f%%', labels=['Death', 'Survival'],
          shadow=True, explode=[0, 0.1], textprops=dict(size=13), colors=['#95A1AD', '#C8CFD7'])
axes2.set_title('Male Survival Rate')
plt.show()

# 4.4 不同船舱等级乘客生还几率
n_Pclass = titanic['Pclass'].value_counts()
print(n_Pclass)
plt.pie(n_Pclass, autopct='%.2f%%', labels=['Pclass 3', 'Pclass 1', 'Pclass 2'], shadow=True,
        explode=[0, 0.1, 0.2], textprops=dict(size=13), colors=['#A0A8C8', '#D0D8E8', '#F3F0F1'])
plt.title('Pclass Ratio')
plt.show()
# 不同舱位的比例

Pclass_count = titanic.groupby(by='Pclass')['Survived'].value_counts()
print(Pclass_count)
plt.figure(figsize=(3 * 5, 5))
axes1 = plt.subplot(1, 3, 1)
axes1.pie(Pclass_count.loc[1], autopct='%.2f%%', labels=['Survival', 'Death'],
          shadow=True, explode=[0, 0.1], textprops=dict(size=13), colors=['#E79482', '#F6CCB3'])
axes1.set_title('Survivable of Pclass 1')
axes2 = plt.subplot(1, 3, 2)
axes2.pie(Pclass_count.loc[2], autopct='%.2f%%', labels=['Death', 'Survival'],
          shadow=True, explode=[0, 0.1], textprops=dict(size=13), colors=['#95A1AD', '#C8CFD7'])
axes2.set_title('Survivable of Pclass 2')
axes3 = plt.subplot(1, 3, 3)
axes3.pie(Pclass_count.loc[3], autopct='%.2f%%', labels=['Death', 'Survival'],
          shadow=True, explode=[0, 0.1], textprops=dict(size=13), colors=['#DFB4A9', '#F1D2CA'])
axes3.set_title('Survivable of Pclass 3')
plt.show()

# 4.5 不同年龄段乘客生还几率
age_range = titanic['Age']
print('Minimum Age:', age_range.min(), 'Maximum Age', age_range.max())
sns.violinplot(x='Survived', y='Age', data=titanic, split=True, palette=['#EAC3D1', '#F8DAD0'])
plt.show()

# 4.6 舱位、性别、生还者数量、死亡者数量之间的关系
plt.subplots(figsize=(12, 8), dpi=80)
sns.violinplot('Sex', 'Pclass', hue='Survived', data=titanic, split=True, palette=['#EAC3D1', '#F8DAD0'])
plt.title('Pclass and Age vs Survived')
plt.yticks(range(0, 4, 1))
plt.show()

# 4.7 不同登入港口乘客的生还几率
embarked_count = titanic.groupby(by='Embarked')['Survived'].value_counts()
print(embarked_count)
sns.factorplot('Embarked', 'Survived', data=titanic, size=3, aspect=2, color='#E79482')
plt.show()
# 不同登入港口乘客的生还情况

plt.figure(figsize=(3 * 5, 5))
axes1 = plt.subplot(1, 3, 1)
axes1.pie(embarked_count.loc['C'][::-1], autopct='%.2f%%', labels=['Death', 'Survival'],
          shadow=True, explode=[0, 0.1], textprops=dict(size=13), colors=['#E79482', '#F6CCB3'], startangle=45)
axes1.set_title('Survivable of Embarked C')
axes2 = plt.subplot(1, 3, 2)
axes2.pie(embarked_count.loc['Q'], autopct='%.2f%%', labels=['Death', 'Survival'],
          shadow=True, explode=[0, 0.1], textprops=dict(size=13), colors=['#95A1AD', '#C8CFD7'])
axes2.set_title('Survivable of Embarked Q')
axes3 = plt.subplot(1, 3, 3)
axes3.pie(embarked_count.loc['S'], autopct='%.2f%%', labels=['Death', 'Survival'],
          shadow=True, explode=[0, 0.1], textprops=dict(size=13), colors=['#DFB4A9', '#F1D2CA'])
axes3.set_title('Survivable of Embarked S')
plt.show()

# 4.8 不同称谓乘客的生还几率
titanic['Title'] = titanic['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(titanic['Title'], titanic['Sex'])
# 提取称谓

plt.figure(figsize=(20, 5))
sns.countplot(x='Title', hue='Survived', data=titanic, palette=['#EAC3D1', '#F8DAD0'])
plt.show()

# 4.9 不同亲友数量乘客的生还几率
sns.countplot(x='SibSp', hue='Survived', data=titanic, palette=['#EAC3D1', '#F8DAD0'])
plt.show()

# 4.10 不同父母子女数量乘客的生还几率
sns.countplot(x='Parch', hue='Survived', data=titanic, palette=['#EAC3D1', '#F8DAD0'])
plt.show()

# 5. 各列间的相关性
for i in ['Sex', 'Embarked']:
    labels = titanic[i].unique().tolist()
    titanic[i] = titanic[i].apply(lambda x: labels.index(x))
cor = titanic.corr()
sns.heatmap(cor, cmap='OrRd')
plt.show()

# 6. 乘客姓名的词云图
x = ''
for i in titanic['Name']:
    x = x + str(i) + ','
print(x)
for i in ['Mrs.', 'Mr.', 'Miss.', 'Master.', 'Dr.', 'Dona.']:
    x = x.replace(i, '')
print(x)
wc = WordCloud(width=400,
               height=400,
               margin=2,
               ranks_only=None,
               prefer_horizontal=0.9,
               mask=None,
               scale=2,
               color_func=None,
               max_words=100,
               min_font_size=4,
               stopwords=None,
               random_state=None,
               background_color='white',
               max_font_size=None,
               font_step=3,
               mode='RGB',
               relative_scaling='auto',
               regexp=None,
               collocations=True,
               colormap='Reds',
               normalize_plurals=True,
               contour_width=0,
               contour_color='black',
               repeat=False)

wc.generate_from_text(x)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.show()

# 7. 聚类
titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)

for i in ['Sex', 'Embarked']:
    labels = titanic[i].unique().tolist()
    titanic[i] = titanic[i].apply(lambda x: labels.index(x))

model = KMeans(n_clusters=5)
kmeans = titanic.drop(columns=['Name', 'Survived', 'Title'])
kmeans = preprocessing.scale(kmeans)
model.fit(kmeans)

# 查看聚类结果
kmeans_c = model.cluster_centers_  # 聚类中心
print('各类聚类中心为:')
print(kmeans_c)
kmeans_labels = model.labels_  # 样本的类别标签
print('各样本的类别标签为:')
print(kmeans_labels)
r1 = pd.Series(model.labels_).value_counts()  # 统计不同类别样本的数目
print('最终每个类别的数目为:')
print(r1)

# 输出聚类分群的结果
cluster_center = pd.DataFrame(model.cluster_centers_, columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
cluster_center.index = pd.DataFrame(model.labels_).drop_duplicates().iloc[:, 0]
print(cluster_center)

# 客户分群雷达图
legen = ['' + str(i + 1) for i in cluster_center.index]
lstype = ['-', ':', '-.']
kinds = list(cluster_center.iloc[:, 0])
print(kinds)

# 由于雷达图要保证数据闭合，因此再添加首列，并转换为 np.ndarray
cluster_center = pd.concat([cluster_center, cluster_center[['Pclass']]], axis=1)
centers = np.array(cluster_center.iloc[:, 0:])
print('Centers')
print(centers)

labels = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
n = len(labels)
angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
angle = np.concatenate((angle, [angle[0]]))
print(angle)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, polar=True)
plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']
plt.rcParams['axes.unicode_minus'] = False

for i in range(len(kinds)):
    ax.plot(angle, centers[i],  linewidth=2, label=kinds[i])  # linestyle=lstype[i]
    ax.fill(angle, centers[i], alpha=0.25)

# plt.thetagrids(range(0, 360, int(360/len(labels))), (labels))
plt.legend(legen)
plt.show()
