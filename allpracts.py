# // pract 1

!pip install pandas
!pip install numpy
!pip install beautifulsoup4
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
url="https://en.wikipedia.org/wiki/List_of_Asian_countries_by_area"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/573.36'}
import requests
response=requests.get(url,headers=headers)
page=response.text
print(page)
soup=BeautifulSoup(page,"html.parser")
table=soup.find("table")
print(table)
SrNo=[]
Country=[]
Area=[]
rows=table.find("tbody").find_all("tr")
for row in rows:
    cells=row.find_all("td")
    if cells:
        SrNo.append(cells[0].get_text().strip("\n"))
        Country.append(cells[1].get_text().strip("\xa0").strip("\n"))
        Area.append(cells[3].get_text().strip("\n").replace(",",""))

print(SrNo)
df=pd.DataFrame()
df["SrNo"]=SrNo
df["Country"]=Country
df["Area"]=Area
df.head(10)
import json
url="https://jsonplaceholder.typicode.com/users"
from urllib.request import urlopen
page=urlopen(url)
data=json.loads(page.read())
Id=[]
Username=[]
Email=[]
for item in data:
    if "id" in item.keys():
        Id.append(item['id'])
    else:
        Id.append("NA")
    if "username" in item.keys():
        Username.append(item['username'])
    else:
        Username.append("NA")
    if "email" in item.keys():
        Email.append(item['email'])
    else:
        Email.append("NA")
df=pd.DataFrame()
df["Id"]=Id
df["Username"]=Username
df["Email"]=Email
df.head(10)
df.info()


# //Pract 2

!pip install seaborn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
titanic_df=pd.read_csv("Titanic Dataset.csv")
titanic_df.head()
titanic_df.info()
titanic_df.isnull().sum()
titanic_df.describe()
titanic_cleaned=titanic_df.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis=1)
titanic_cleaned.head()
titanic_cleaned.info()
titanic_cleaned["Age"]=titanic_cleaned["Age"].fillna(titanic_cleaned.groupby("Pclass")["Age"].transform("mean"))
titanic_cleaned.isnull().sum()
sns.catplot(x="Sex",hue="Survived",kind="count",data=titanic_cleaned)
titanic_cleaned.groupby(['Sex','Survived'])['Survived'].count()
group1=titanic_cleaned.groupby(['Sex','Survived'])
gender_survived=group1.size().unstack()
sns.heatmap(gender_survived,annot=True,fmt="d")
group2=titanic_cleaned.groupby(['Pclass','Survived'])
pclass_survived=group2.size().unstack()
sns.heatmap(pclass_survived,annot=True,fmt="d")
sns.violinplot(x="Sex",y="Age",hue="Survived",data=titanic_cleaned,split=True)
titanic_corr=titanic_cleaned.drop(['Sex','Embarked'],axis=1)
titanic_corr.corr(method="pearson")
sns.heatmap(titanic_corr.corr(method="pearson"),annot=True,vmax=1)



# //Pract 3
import sys 
!{sys.executable} -m pip install pandas seaborn matplotlib numpy 
import os 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 
print(os.getcwd()) 
df = pd.read_csv("MTCARS.csv") 
print(df.head()) 
print(df.tail()) 
df.info() 
print(df.isnull().sum()) 
print(df.describe()) 
print(df.shape) 
df_cleaned = df.drop(['model'], axis=1) 
df_cleaned["mpg"] = df_cleaned["mpg"].fillna(df_cleaned.groupby("cyl")["mpg"].transform("mean")) 
plt.figure(figsize=(8, 5)) 
sns.countplot(x="cyl", data=df_cleaned) 
plt.show() 
plt.figure(figsize=(8, 5)) 
plt.hist(df_cleaned['mpg'], bins=10, color='skyblue', edgecolor='black') 
plt.show() 
plt.figure(figsize=(8, 5)) 
sns.boxplot(x=df_cleaned['hp']) 
plt.show() 
sns.catplot(x="cyl", hue="am", kind="count", data=df_cleaned) 
plt.show() 
group1 = df_cleaned.groupby(['cyl', 'am']) 
cyl_am_table = group1.size().unstack() 
sns.heatmap(cyl_am_table, annot=True, fmt="d", cmap="YlGnBu") 
plt.show() 
plt.figure(figsize=(10, 6)) 
sns.violinplot(x="cyl", y="mpg", hue="am", data=df_cleaned, split=True) 
plt.show() 
plt.figure(figsize=(10, 6)) 
sns.scatterplot(x='wt', y='mpg', hue='cyl', size='hp', data=df_cleaned) 
plt.show() 
mtcars_corr = df_cleaned.corr(method="pearson") 
print(mtcars_corr) 
plt.figure(figsize=(12, 8)) 
sns.heatmap(mtcars_corr, annot=True, vmax=1, vmin=-1, cmap='coolwarm') 
plt.show() 


# //pract 4
import numpy as np 
from sklearn import datasets 
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt 
x, y, coef = datasets.make_regression( 
n_samples=100, 
n_features=1, 
n_informative=1, 
noise=10, 
coef=True, 
random_state=0 
) 
x = np.interp(x, (x.min(), x.max()), (0, 20)) 
y = np.interp(y, (y.min(), y.max()), (20000, 150000)) 
plt.plot(x, y, '.', label='Training Data') 
plt.xlabel("Years of Experience") 
plt.ylabel("Salary") 
plt.title("Experience vs Salary") 
plt.legend() 
plt.show() 


reg_model = LinearRegression() 
reg_model.fit(x, y) 
y_predicted = reg_model.predict(x) 
plt.plot(x, y, '.', label='Training Data') 
plt.plot(x, y_predicted, '.', color='red', label='Predicted Data') 
plt.xlabel("Years of Experience") 
plt.ylabel("Salary") 
plt.title("Experience vs Salary") 
plt.legend() 
plt.show() 


from sklearn.linear_model import LinearRegression 
reg_model = LinearRegression() 
x = np.random.rand(100, 1) 
error = np.random.rand(100, 1) 
b0 = 10 
b1 = 7 
y = b0 + b1 * x + error 
reg_model.fit(x, y) 
y_predicted = reg_model.predict(x) 
plt.plot(x, y_predicted, ".", color="black", label="Predicted Data") 
plt.plot(x, y, ".", label="Training Data") 
plt.xlabel("X") 
plt.ylabel("Y") 
plt.title("X vs Y") 
plt.legend() 


# //pract 5
import pandas as pd
import matplotlib.pyplot as plt
!pip install scikit-learn
import sklearn
boston_df = pd.read_csv("Boston.csv")
boston_df.head()
boston_df.info()
boston_df = boston_df.drop("Unnamed: 0",axis = 1)
boston_df.info()
boston_x = pd.DataFrame(boston_df.iloc[:,:13])
boston_y = pd.DataFrame(boston_df.iloc[:,-1])
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(boston_x,boston_y,test_size=0.3)
print(f'X Train Size: {X_train.shape}')
print(f'X Test Size: {X_test.shape}')
print(f'Y Train Size: {Y_train.shape}')
print(f'Y Test Size: {Y_test.shape}')
from sklearn.linear_model import LinearRegression
reg_model = LinearRegression()
reg_model.fit(X_train,Y_train)
y_predicted = reg_model.predict(X_test)
Y_pred = pd.DataFrame(y_predicted,columns=["Predicted_Y"])
Y_pred.head()

plt.scatter(Y_test,Y_pred,c="green")
plt.xlabel("Actual Price(medv)")
plt.ylabel("Predicted Price")
plt.show()


# //pract 6 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
breast_cancer_data = load_breast_cancer()
X_df = pd.DataFrame(breast_cancer_data.data,columns=breast_cancer_data.feature_names)
X_df.head()
X_df.info()
X_df=X_df[['mean area','mean compactness']]     ### Method for redefining the DataFrame
X_df.head()
X_df.info()
Y_df = pd.Categorical.from_codes(breast_cancer_data.target,breast_cancer_data.target_names)
print(Y_df)
Y_df = pd.get_dummies(Y_df,drop_first = True)
Y_df.info()
print(Y_df)
X_train,X_test,Y_train,Y_test = train_test_split(X_df,Y_df,random_state=1,test_size=0.25,shuffle=True)
X_test.info()
Y_test.info()
X_train.info()
Y_train.info()
# KNN model
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn.fit(X_train, Y_train)

combined_df=pd.concat([X_test,Y_test],axis=1)
sns.scatterplot(x="mean area",y="mean compactness",hue="benign",data=combined_df)
Y_pred=knn.predict(X_test)

plt.scatter(X_test["mean area"],X_test["mean compactness"],c=Y_pred,cmap="coolwarm",alpha=0.7)
cf=confusion_matrix(Y_test,Y_pred)
print(cf)
tp,fn,fp,tn=confusion_matrix(Y_test,Y_pred,labels=[1,0]).reshape(-1)
print(tp,fn,fp,tn)
labels=["True Negatives","False Positive","False Negative","True Positive"]
labels=np.asarray(labels).reshape(2,2)

categories=["Zero","One"]
ax=plt.subplot()
sns.heatmap(cf,annot=True,ax=ax)
ax.set_xlabel

categories=["Zero","One"]
ax=plt.subplot()
sns.heatmap(cf,annot=True,ax=ax)
ax.set_xlabel("Predicted Values")
ax.set_ylabel("Actual Values")
ax.set_title("Confussion Matrix")
ax.xaxis.set_ticklabels(["Malignant","Benign"])
ax.yaxis.set_ticklabels(["Malignant","Benign"])

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# //pract 7
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
titanic_df=pd.read_csv("train.csv")
titanic_df.info()
features=['Pclass','Sex','Age','SibSp','Parch']
x=titanic_df[features]
x.info()
Y=titanic_df['Survived']
Y.info()
x['Sex']=x['Sex'].map({'male':0,'female':1})
x.head()
x['Age'].fillna(x['Age'].median(),inplace=True)
x.isnull().sum()
X_train,X_test,Y_train,Y_test=train_test_split(x,Y,test_size=0.2,random_state=10)
dtmodel=DecisionTreeClassifier(criterion='entropy',max_depth=4,random_state=10)

dtmodel.fit(X_train,Y_train)
Y_pred=dtmodel.predict(X_test)
print("Accuracy: ",accuracy_score(Y_test,Y_pred))

print(classification_report(Y_test,Y_pred))
plt.figure(figsize=(18,10))
plot_tree(dtmodel,feature_names=features,class_names=['Not Survived','Survived'],filled=True)



# //pract 8
// Switch or create database
use myDatabase

// Check databases
show dbs

// Create collection
db.createCollection("students")

// Show collections
show collections

db.students.insertOne({
  name: "John",
  age: 22,
  course: "CS",
  marks: 85
})

db.students.insertMany([
  { name: "Alice", age: 20, course: "IT", marks: 78 },
  { name: "Bob", age: 23, course: "CS", marks: 92 },
  { name: "Charlie", age: 21, course: "AI", marks: 88 }
])

db.students.find()

db.students.find({ course: "CS" })

db.students.updateOne(
  { name: "John" },
  { $set: { marks: 90 } }
)
db.students.updateMany(
  { course: "CS" },
  { $inc: { marks: 5 } }
)
db.students.find({ marks: { $gt: 80 } })   // greater than
db.students.find({ marks: { $lt: 90 } })   // less than
db.students.find({ marks: { $gte: 85 } })  // >=
db.students.find({ marks: { $lte: 90 } })  // <=

db.students.find({
  $and: [{ course: "CS" }, { marks: { $gt: 80 } }]
})

db.students.find({
  $or: [{ course: "AI" }, { marks: { $lt: 80 } }]
})


// Delete one
db.students.deleteOne({ name: "Alice" })

// Delete many
db.students.deleteMany({ course: "IT" })

// Include only name and marks
db.students.find({}, { name: 1, marks: 1, _id: 0 })
// Ascending
db.students.find().sort({ marks: 1 })

// Descending
db.students.find().sort({ marks: -1 })

db.students.aggregate([
  { $match: { marks: { $gt: 80 } } }
])


db.students.aggregate([
  {
    $project: {
      name: 1,
      grade: {
        $cond: { if: { $gte: ["$marks", 85] }, then: "A", else: "B" }
      }
    }
  }
])

db.students.aggregate([
  {
    $group: {
      _id: "$course",
      totalMarks: { $sum: "$marks" },
      avgMarks: { $avg: "$marks" }
    }
  }
])

db.students.mapReduce(
  function() {
    emit(this.course, this.marks);
  },
  function(key, values) {
    return Array.sum(values);
  },
  {
    out: "course_totals"
  }
)




//Mongo PDF 


// ======================
// DATABASE & COLLECTION
// ======================

use myDB

use myNewDB
db.myNewCollection1.insertOne({ x: 1 })

db.myNewCollection2.insertOne({ x: 1 })

db.myNewCollection3.createIndex({ y: 1 })


// ======================
// INVENTORY DATA (INSERT)
// ======================

db.inventory.insertMany([
 { item: "journal", qty: 25, size: { h: 14, w: 21, uom: "cm" }, status: "A" },
 { item: "notebook", qty: 50, size: { h: 8.5, w: 11, uom: "in" }, status: "P" },
 { item: "paper", qty: 100, size: { h: 8.5, w: 11, uom: "in" }, status: "D" },
 { item: "planner", qty: 75, size: { h: 22.85, w: 30, uom: "cm" }, status: "D" },
 { item: "postcard", qty: 45, size: { h: 10, w: 15.25, uom: "cm" }, status: "A" }
])


// ======================
// READ / FIND QUERIES
// ======================

// Select all
db.inventory.find({})

// Equality
db.inventory.find({ status: "D" })

// IN operator
db.inventory.find({ status: { $in: ["A", "D"] } })

// AND condition
db.inventory.find({ status: "A", qty: { $lt: 30 } })

// OR condition
db.inventory.find({
  $or: [{ status: "A" }, { qty: { $lt: 30 } }]
})

// AND + OR
db.inventory.find({
  status: "A",
  $or: [{ qty: { $lt: 30 } }, { item: /^p/ }]
})


// ======================
// OPERATORS EXAMPLES
// ======================

db.mycol.find({"by":"tutorials point"}).pretty()

db.mycol.find({"likes":{$lt:50}}).pretty()

db.mycol.find({"likes":{$lte:50}}).pretty()

db.mycol.find({"likes":{$gt:50}}).pretty()

db.mycol.find({"likes":{$gte:50}}).pretty()

db.mycol.find({"likes":{$ne:50}}).pretty()

db.mycol.find({"name":{$in:["Raj","Ram","Raghu"]}}).pretty()

db.mycol.find({"name":{$nin:["Ramu","Raghav"]}}).pretty()


// ======================
// UPDATE OPERATIONS
// ======================

db.mycol.update(
  {'title':'MongoDB Overview'},
  {$set:{'title':'New MongoDB Tutorial'}}
)

db.mycol.find()

// SAVE (REPLACE)
db.mycol.save({
  "_id": ObjectId("5983548781331adf45ec7"),
  "title": "Tutorials Point New Topic",
  "by": "Tutorials Point"
})

db.mycol.find()


// ======================
// DELETE OPERATIONS
// ======================

// Remove specific
db.mycol.remove({'title':'MongoDB Overview'})

db.mycol.find()

// Remove one
db.COLLECTION_NAME.remove(DELETION_CRITERIA,1)

// Remove all
db.mycol.remove()

db.mycol.find()


// ======================
// PROJECTION
// ======================

db.mycol.find({},{"title":1,_id:0})


// ======================
// SORTING
// ======================

db.mycol.find({},{"title":1,_id:0}).sort({"title":-1})


// ======================
// MAP REDUCE
// ======================

// Map function
var mapFunction1 = function() {
  emit(this.cust_id, this.price);
};

// Reduce function
var reduceFunction1 = function(keyCustId, valuesPrices) {
  return Array.sum(valuesPrices);
};

// Execute mapReduce
db.orders.mapReduce(
  mapFunction1,
  reduceFunction1,
  { out: "map_reduce_example" }
);

// View result
db.map_reduce_example.find().sort({ _id: 1 })


// ======================
// ORDERS DATA (MAP REDUCE)
// ======================

db.orders.insertMany([
 { _id: 1, cust_id: "Ant O. Knee", ord_date: new Date("2020-03-01"), price: 25, items: [ { sku: "oranges", qty: 5, price: 2.5 }, { sku: "apples", qty: 5, price: 2.5 } ], status: "A" },
 { _id: 2, cust_id: "Ant O. Knee", ord_date: new Date("2020-03-08"), price: 70, items: [ { sku: "oranges", qty: 8, price: 2.5 }, { sku: "chocolates", qty: 5, price: 10 } ], status: "A" },
 { _id: 3, cust_id: "Busby Bee", ord_date: new Date("2020-03-08"), price: 50, items: [ { sku: "oranges", qty: 10, price: 2.5 }, { sku: "pears", qty: 10, price: 2.5 } ], status: "A" },
 { _id: 4, cust_id: "Busby Bee", ord_date: new Date("2020-03-18"), price: 25, items: [ { sku: "oranges", qty: 10, price: 2.5 } ], status: "A" },
 { _id: 5, cust_id: "Busby Bee", ord_date: new Date("2020-03-19"), price: 50, items: [ { sku: "chocolates", qty: 5, price: 10 } ], status: "A"},
 { _id: 6, cust_id: "Cam Elot", ord_date: new Date("2020-03-19"), price: 35, items: [ { sku: "carrots", qty: 10, price: 1.0 }, { sku: "apples", qty: 10, price: 2.5 } ], status: "A" },
 { _id: 7, cust_id: "Cam Elot", ord_date: new Date("2020-03-20"), price: 25, items: [ { sku: "oranges", qty: 10, price: 2.5 } ], status: "A" },
 { _id: 8, cust_id: "Don Quis", ord_date: new Date("2020-03-20"), price: 75, items: [ { sku: "chocolates", qty: 5, price: 10 }, { sku: "apples", qty: 10, price: 2.5 } ], status: "A" },
 { _id: 9, cust_id: "Don Quis", ord_date: new Date("2020-03-20"), price: 55, items: [ { sku: "carrots", qty: 5, price: 1.0 }, { sku: "apples", qty: 10, price: 2.5 }, { sku: "oranges", qty: 10, price: 2.5 } ], status: "A" },
 { _id: 10, cust_id: "Don Quis", ord_date: new Date("2020-03-23"), price: 25, items: [ { sku: "oranges", qty: 10, price: 2.5 } ], status: "A" }
])
