#import modules

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, r2_score
import time
from matplotlib import pyplot as plt

#Create your df here:

df = pd.read_csv("profiles.csv")

#Examining the data
#print df.head()

#Examining the distribution of ages
"""plt.hist(df.age, bins=30)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
plt.show()"""

# Examining the distribution of incomes
"""plt.hist(df[df['income']!=-1].income, bins=50)
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.xlim(0, 1000000)
plt.show()"""

# Examining the unique values of certain columns
"""print df.body_type.value_counts()
print df.diet.value_counts()
print df.smokes.value_counts()
print df.drugs.value_counts()
print df.drinks.value_counts()
print df.education.value_counts()"""

# Creating new columns for some categorised data

# For body type, the mapping identifies those where people identify as being athletic, fit, or jacked
body_type_mapping = {"average": 0, "fit": 1, "athletic": 1, "thin": 0, "curvy": 0, "a little extra": 0, "skinny": 0,
                     "full figured": 0, "overweight": 0, "jacked": 1, "used up": 0, "rather not say": 0}
df["body_type_code"] = df.body_type.map(body_type_mapping)

drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df["drinks_code"] = df.drinks.map(drink_mapping)

smokes_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "yes": 3, "trying to quit": 4}
df["smokes_code"] = df.smokes.map(smokes_mapping)

drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}
df["drugs_code"] = df.drugs.map(drugs_mapping )

diet_anything_mapping = {"mostly anything": 1, "anything": 2, "strictly anything": 3,
                         "mostly other": 0, "other": 0, "strictly other": 0,
                         "mostly vegetarian": 0, "vegetarian": 0, "strictly vegetarian": 0,
                         "mostly vegan": 0, "vegan": 0, "strictly vegan": 0,
                         "mostly kosher": 0, "kosher": 0, "strictly kosher": 0,
                         "mostly halal": 0, "halal": 0, "strictly halal": 0}

diet_other_mapping = {"mostly anything": 0, "anything": 0, "strictly anything": 0,
                      "mostly other": 1, "other": 2, "strictly other": 3,
                      "mostly vegetarian": 0, "vegetarian": 0, "strictly vegetarian": 0,
                      "mostly vegan": 0, "vegan": 0, "strictly vegan": 0,
                      "mostly kosher": 0, "kosher": 0, "strictly kosher": 0,
                      "mostly halal": 0, "halal": 0, "strictly halal": 0}

diet_vegetarian_mapping = {"mostly anything": 0, "anything": 0, "strictly anything": 0,
                           "mostly other": 0, "other": 0, "strictly other": 0,
                           "mostly vegetarian": 1, "vegetarian": 2, "strictly vegetarian": 3,
                           "mostly vegan": 0, "vegan": 0, "strictly vegan": 0,
                           "mostly kosher": 0, "kosher": 0, "strictly kosher": 0,
                           "mostly halal": 0, "halal": 0, "strictly halal": 0}

diet_vegan_mapping = {"mostly anything": 0, "anything": 0, "strictly anything": 0,
                      "mostly other": 0, "other": 0, "strictly other": 0,
                      "mostly vegetarian": 0, "vegetarian": 0, "strictly vegetarian": 0,
                      "mostly vegan": 1, "vegan": 2, "strictly vegan": 3,
                      "mostly kosher": 0, "kosher": 0, "strictly kosher": 0,
                      "mostly halal": 0, "halal": 0, "strictly halal": 0}

diet_kosher_mapping = {"mostly anything": 0, "anything": 0, "strictly anything": 0,
                       "mostly other": 0, "other": 0, "strictly other": 0,
                       "mostly vegetarian": 0, "vegetarian": 0, "strictly vegetarian": 0,
                       "mostly vegan": 0, "vegan": 0, "strictly vegan": 0,
                       "mostly kosher": 1, "kosher": 2, "strictly kosher": 3,
                       "mostly halal": 0, "halal": 0, "strictly halal": 0}

diet_halal_mapping = {"mostly anything": 0, "anything": 0, "strictly anything": 0,
                      "mostly other": 0, "other": 0, "strictly other": 0,
                      "mostly vegetarian": 0, "vegetarian": 0, "strictly vegetarian": 0,
                      "mostly vegan": 0, "vegan": 0, "strictly vegan": 0,
                      "mostly kosher": 0, "kosher": 0, "strictly kosher": 0,
                      "mostly halal": 1, "halal": 2, "strictly halal": 3}

df["diet_anything_code"] = df.diet.map(diet_anything_mapping)
df["diet_other_code"] = df.diet.map(diet_other_mapping)
df["diet_vegetarian_code"] = df.diet.map(diet_vegetarian_mapping)
df["diet_vegan_code"] = df.diet.map(diet_vegan_mapping)
df["diet_kosher_code"] = df.diet.map(diet_kosher_mapping)
df["diet_halal_code"] = df.diet.map(diet_halal_mapping)

# For education, as well as categorising the responses, an effort has been made to qualitatively quantify the "level"
# of education associated with each category
education_mapping = {"graduated from ph.d program": 5, "ph.d program": 5, "working on ph.d program": 4,
                     "dropped out of ph.d program": 4,
                     "graduated from masters program": 4, "masters program": 4, "working on masters program": 3,
                     "dropped out of masters program": 3,
                     "graduated from law school": 4, "law school": 4, "working on law school": 3,
                     "dropped out of law school": 3,
                     "graduated from med school": 4, "med school": 4, "working on med school": 3,
                     "dropped out of med school": 3,
                     "graduated from college/university": 3, "college/university": 3, "working on college/university": 1,
                     "dropped out of college/university": 1,
                     "graduated from two-year college": 2, "two-year college": 2, "working on two-year college": 1,
                     "dropped out of two-year college": 1,
                     "graduated from high school": 1, "high school": 1, "working on high school": 0,
                     "dropped out of high school": 0,
                     "graduated from space camp": 0, "space camp": 0, "working on space camp": 0,
                     "dropped out of space camp": 0}

df["education_code"] = df.education.map(education_mapping)

# In line with the instructions, the essay responses have been processed.
essay_cols = ["essay0", "essay1", "essay2", "essay3", "essay4", "essay5", "essay6", "essay7", "essay8", "essay9"]

# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
# Converting to lower case
all_essays = all_essays.apply(lambda x: x.lower())
# Finding the length of the total essays
df["essay_len"] = all_essays.apply(lambda x: len(x))
# Counting the number of times responses make allusions to books or reading
df["essay_reading"] = all_essays.apply(lambda x: x.count('book')) \
                      + all_essays.apply(lambda x: x.count('read')) \

# Examining the distribution of essay lengths
"""plt.hist(df.essay_len, bins=1000)
plt.xlabel("Essay Length")
plt.ylabel("Frequency")
plt.xlim(0, 20000)
plt.ylim(0, 2000)
plt.show()"""

# Examining the distribution of mentions of reading
"""plt.hist(df.essay_reading, bins=60)
plt.xlabel("Essay mentions of Reading")
plt.ylabel("Frequency")
plt.xlim(0, 20)
plt.ylim(0, 20000)
plt.show()"""

# The first question is, can we classify whether someone identifies as fit, athletic or jacked based on some information
# about their lifestyle, as well as their age and education level
# The relevant data columns are defined, and a dataframe constructed
question1_columns = ["body_type_code", "smokes_code", "drinks_code", "drugs_code", "age",
                     "diet_anything_code", "diet_other_code", "diet_vegetarian_code",
                     "diet_vegan_code", "diet_kosher_code", "diet_halal_code",
                     "education_code"]

question1_data = df[question1_columns]

# For rows where any values are null, the data is dropped.
question1_data = question1_data.dropna(axis=0, how='any', subset=question1_columns)

# The data, except the first category column, is normalised
question1_data_unscaled = question1_data.values[:, 1:]
min_max_scaler = preprocessing.MinMaxScaler()
question1_data_scaled = min_max_scaler.fit_transform(question1_data_unscaled)

# the label and feature data is defined
labels = pd.DataFrame(question1_data, columns=question1_data.columns[[0]]).values.ravel()
features = pd.DataFrame(question1_data_scaled , columns=question1_data.columns[1:]).values

# the data is split into testing and training sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                            test_size=0.33, random_state=42)
# the first classifider, a K Nearest Neighbors algorithm, is trained. With trial and error, a value of k=5 is chosen
start = time.clock()
classifier1 = KNeighborsClassifier(n_neighbors=5)
classifier1.fit(features_train, labels_train)
labels_guess1 = classifier1.predict(features_test)
print(accuracy_score(labels_test, labels_guess1),
      recall_score(labels_test, labels_guess1),
      precision_score(labels_test, labels_guess1),
      f1_score(labels_test, labels_guess1),
      time.clock()-start)

# Secondly, a support vector machine is trained, using an RBF kernel
start = time.clock()
classifier2 = SVC(kernel="rbf")
classifier2.fit(features_train, labels_train)
labels_guess2 = classifier2.predict(features_test)
print(accuracy_score(labels_test, labels_guess2),
      recall_score(labels_test, labels_guess2),
      precision_score(labels_test, labels_guess2),
      f1_score(labels_test, labels_guess2),
      time.clock()-start)

# Third, a Naive Bayes classifier is trained
start = time.clock()
classifier3 = MultinomialNB()
classifier3.fit(features_train, labels_train)
labels_guess3 = classifier3.predict(features_test)
print(accuracy_score(labels_test, labels_guess3),
      recall_score(labels_test, labels_guess3),
      precision_score(labels_test, labels_guess3),
      f1_score(labels_test, labels_guess3),
      time.clock()-start)

# In each case, the accuracy, recall, precision and f1 score of each classification model is recorded.

# Question 2 is, can we predict someone's education level, based on some information about their age and income,

question2_columns = ["age", "education_code", "income",
                     "essay_len", "essay_reading",
                     "drinks_code", "smokes_code", "drugs_code", "diet_vegetarian_code", "diet_vegan_code"]

question2_data = df[question2_columns]

# all of the NA rows are dropped, as are the rows for which income is set equal to -1
question2_data = question2_data.dropna(axis=0, how='any', subset=question2_columns)
df = df[df.income != -1]

# outputs and inputs are defined
outputs = pd.DataFrame(question2_data, columns=question2_data.columns[[0]]).values.ravel()
inputs = pd.DataFrame(question2_data, columns=question2_data.columns[1:]).values

# data is split into testing and training sets
inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs,
                                                                          test_size=0.2, random_state=42)
# the first model is a Linear Regression
start = time.clock()
regression1 = LinearRegression()
regression1.fit(inputs_train, outputs_train)
outputs_guess1 = regression1.predict(inputs_test)
print r2_score(outputs_test, outputs_guess1), time.clock()-start

# The second is a K Nearest Neighbors Regression. Uniform weights are used, as these are found to give a slightly better
# fit
start = time.clock()
regression2 = KNeighborsRegressor(n_neighbors=5, weights="uniform")
regression2.fit(inputs_train, outputs_train)
outputs_guess2 = regression2.predict(inputs_test)
print r2_score(outputs_test, outputs_guess2), time.clock()-start

# for both regression models, the R**2 score is printed.

