import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = sns.load_dataset('iris')
df.head()

encoder = LabelEncoder()
df['species'] = encoder.fit_transform(df['species'])
df.head()

df = df[['sepal_length','petal_length','species']]
df.head()

X = df.iloc[:,0:2]
y = df.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

clf = LogisticRegression(multi_class='multinomial')  # multinomial function is called softmax function

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test,y_pred))

pd.DataFrame(confusion_matrix(y_test,y_pred))

# prediction
query = np.array([[3.4,2.7]])
clf.predict_proba(query)

clf.predict(query)

from mlxtend.plotting import plot_decision_regions  # plot decision boundaries

plot_decision_regions(X.values, y.values, clf, legend=2)

# Adding axes annotations
plt.xlabel('sepal length [cm]')
plt.xlabel('petal length [cm]')
plt.title('Softmax on Iris')

plt.show()

 