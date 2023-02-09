import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("PlayPredictor.csv")

# print(data.head())



# Clean, Prepare and manipulate data
feature_nm=['Whether','Temperature']

print("Feture name",feature_nm)

# Creating labelEncoder
label_encoder = preprocessing.LabelEncoder()

data['Play'] = label_encoder.fit_transform(data['Play'])
data['Whether'] = label_encoder.fit_transform(data['Whether'])
data['Temperature'] = label_encoder.fit_transform(data['Temperature'])



# Encode labels in column 'species'.


# Combining weather and temp into single listof tuples
features=list(zip(data['Whether'],data['Temperature']))

#data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.5)

# train data
classifier = KNeighborsClassifier(n_neighbors=3)

classifier.fit(features, data['Play'])

# Test data data_test
predictions = classifier.predict([[0,2]]) # 0:Overcast, 2: Mild
print(predictions)