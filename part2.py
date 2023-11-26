# 3 Training
# 3-1 Logistic regression

# train logistic Regression
LR_classifier = LogisticRegression(random_state= 0)
LR_classifier.fit(X, y_train)
LR_acc_training_data = 100*LR_classifier.score(X, y_train)
print('Logistic Regression : The accuracy on the training data is : ',  LR_acc_training_data, '%')

# 3-2 Random Forest
# train random forest
RF_classifier = RandomForestClassifier(random_state = 0)
RF_classifier.fit(X, y_train)
RF_acc_training_data = 100*RF_classifier.score(X, y_train)
print('Random Forest : The accuracy on the training data is : ',  RF_acc_training_data, '%')

# 3-3 Naive Bayes
# train naive bayes
gnb_classifier = GaussianNB()
gnb_classifier.fit(X, y_train)
gnb_acc_training_data = 100*gnb_classifier.score(X, y_train)
print('Naive Bayes : The accuracy on the training data is : ',  gnb_acc_training_data, '%')

# 3-3 Artificial Neural Networks
# ANN
ann_classifier = tf.keras.models.Sequential()
ann_classifier.add(tf.keras.layers.Dense(units=5, activation='relu'))
ann_classifier.add(tf.keras.layers.Dense(units=4, activation='softmax'))
ann_classifier.compile(optimizer="adam", loss="SparseCategoricalCrossentropy", metrics=["accuracy"])
hist=ann_classifier.fit(X, y_train, epochs = 30)
