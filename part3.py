# 4-Testing
# 4-1 Test data preperation
# prepare test data
# clean
X_test = text_cleaning_function(X_test)

# vecotrise emails (use transform where the vectoriser is fit on training data)
X_test = vectorizer_data.transform(X_test).toarray()

# 4-2 Logistic Regression
# predict test output using the traind model
LR_y_pred = LR_classifier.predict(X_test)

# confusion matrix
LR_cm = confusion_matrix(y_test, LR_y_pred)
print('Logistic Regression : The confusion matrix values are : ', LR_cm)

# accuracy calculation
LR_acc = 100*accuracy_score(y_test, LR_y_pred)
print('Logistic Regression : The accuracy on the test set is : ', LR_acc, '%')

# 4-3 Random Forest
# predict test output using the traind model
RF_y_pred = RF_classifier.predict(X_test)

# confusion matrix
RF_cm = confusion_matrix(y_test, RF_y_pred)
print('Random Forest : The confusion matrix values are : ', RF_cm)

# accuracy calculation
RF_acc = 100*accuracy_score(y_test, RF_y_pred)
print('Random Forest : The accuracy on the test set is : ', RF_acc, '%')

# 4-4 Naive Bayes
# predict test output using the traind model
gnb_y_pred = gnb_classifier.predict(X_test)

# confusion matrix
gnb_cm = confusion_matrix(y_test, gnb_y_pred)
print('Naive Bayes : The confusion matrix values are : ', gnb_cm)

# accuracy calculation
gnb_acc = 100*accuracy_score(y_test, gnb_y_pred)
print('Naive Bayes : The accuracy on the test set is : ', gnb_acc, '%')

# 4-5 Artificial Neural Networks
# ANN
# predict test output using the traind model
ann_y_pred = np.argmax(ann_classifier.predict(X_test), axis = 1)

# confusion matrix
ann_cm = confusion_matrix(y_test, ann_y_pred)
print('Artificial Neural Network : The confusion matrix values are : ', ann_cm)

# accuracy calculation
ann_acc = 100*accuracy_score(y_test, ann_y_pred)
print('Artificial Neural Network : The accuracy on the test set is : ', ann_acc, '%')


