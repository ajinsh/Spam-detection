## Author : Ajinkya Shinde


#### Load the required packages
import csv 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
import matplotlib.pyplot as plt




#### Read the dataset and store it into numpy array
f = open('spambase.data','r')
csvreader = csv.reader(f)
data = list(csvreader)

# Store the data into  numpy array
spam_arr = np.array(data,float)


# Seperate the spam dataset into labels and features 
spam_labels = np.asarray(spam_arr[:,-1],dtype='int')
spam_data = np.delete(spam_arr,57, axis = 1)

# print(spam_data.shape)
# print(spam_labels.shape)


##################################################################
################# Data Processing Begins #########################
##################################################################

#### Split data into ~1/2 training and 1/2 test
train_features, test_features, train_labels, test_labels = train_test_split(spam_data, spam_labels, test_size = 0.5) 

#### Scaled training data features

# train_scaled = preprocessing.scale(train_features)
# print("train_scaled using skscaling",train_scaled)

train_features_mean = np.mean(train_features,axis = 0)
train_features_std = np.std(train_features,axis = 0)


for each in train_features:
	for i in range(57):
		if train_features_std[i] != 0:
			each[i] = (each[i]-train_features_mean[i])/train_features_std[i]
		else:
			each[i] = 0
# print("train_features without using skscaling",train_features)	

#### Scaled testing data features using train data mean and train data std
for each in test_features:
	for i in range(57):
		if train_features_std[i] != 0:
			each[i] = (each[i]-train_features_mean[i])/train_features_std[i]
		else:
			each[i] = 0
# print("test_features without using skscaling",test_features)	

# print(train_features.shape)
# print(train_features)
# print(test_features.shape)
# print(test_features)



# ##################################################################
# ################# Data Processing Ends #########################
# ##################################################################


# ##################################################################
# ################# Experiment 1  #########################
# ##################################################################


model = SVC(kernel='linear')
model.fit(train_features, train_labels)  
test_pred = model.predict(test_features)


# # print(classification_report(test_labels,test_pred))  
# # print(confusion_matrix(test_labels,test_pred))  

# # print(test_labels.shape[0])
# # print(test_pred.shape[0])

# TP = 0
# FP = 0
# TN = 0
# FN = 0


# for i in range(test_labels.shape[0]):
# 	if test_labels[i] == 0:
# 		if test_labels[i] == test_pred[i]:
# 			TP = TP + 1
# 		else:
# 			FN = FN + 1
# 	else:
# 		if test_labels[i] == test_pred[i]:
# 			TN = TN + 1
# 		else:
# 			FP = FP + 1

# print("Recall for non-spam emails    :",round(TP/(TP + FN),2))
# print("Precision for non-spam emails :",round(TP/(TP + FP),2))
# print("Recall for spam emails        :",round(TN/(TN + FP),2))
# print("Precision for spam emails     :",round(TN/(TN + FN),2))
# print("Accuracy:",accuracy_score(test_labels,test_pred))

# fpr, tpr, thresholds = roc_curve(test_labels,model.decision_function(test_features))
# print('fpr',fpr.shape)
# print('tpr',tpr.shape)
# print('thresholds', thresholds.shape)
# plt.plot(fpr,tpr)
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.title('Receiver Operating Characterestic (ROC) Curve')
# plt.legend()
# plt.show()


# ##################################################################
# ################# Experiment 2 ###################################
# ##################################################################


# Co-effient for support vectors - α   
alpha_support_vec = model.dual_coef_


# Support vectors returned by the model
support_vec = model.support_vectors_

# Calculate the weight vector
weight_vec =np.dot(alpha_support_vec, support_vec)

# Get absolute of weight vector
abs_wt_vec = abs(weight_vec[0])
# print(abs_wt_vec)


# Get indices of weight vector for elements in weight vector
# arranged from highest to lowest
idx_ftselec= abs_wt_vec.argsort()[::-1]
# print('idx_ftselec', idx_ftselec)

# Get the top 5 features index
print('Top 5 features are at index ', idx_ftselec[:5])


# Create a new model for experiment 2
model1 = SVC(kernel='linear')
e2_accuracy = []
e2_tot_feat = []
for i in range(57):
	e2_trainf = train_features[:,idx_ftselec[0:i+1]]
	e2_testf = test_features[:,idx_ftselec[0:i+1]]
	if i == 0:
		feat_select = 2
	else:
		feat_select = feat_select + 1
	e2_tot_feat.append(feat_select)
	model1.fit(e2_trainf, train_labels)
	e2_testf_pred  = model1.predict(e2_testf)
	e2_accuracy.append(accuracy_score(test_labels,e2_testf_pred))
	i = i + 1


# Plot the accuracy vs m (no. of features)
# plt.plot(e2_tot_feat,e2_accuracy)
# plt.ylabel('Accuracy')
# plt.xlabel('Number of features')
# plt.title('Experiment 2 - Accuracy vs No. of features')
# plt.show()


# ##################################################################
# ################# Experiment 3 ###################################
# ##################################################################

model2 = SVC(kernel='linear')
e3_accuracy = []
e3_tot_feat = []


for i in range(57):
	rand_idx = np.random.choice(np.arange(57),i+1)
	e3_trainf = train_features[:,rand_idx]
	e3_testf = test_features[:,rand_idx]
	if i == 0:
		feat_select1 = 2
	else:
		feat_select1 = feat_select1 + 1
	e3_tot_feat.append(feat_select1)
	model2.fit(e3_trainf, train_labels)
	e3_testf_pred  = model2.predict(e3_testf)
	e3_accuracy.append(accuracy_score(test_labels,e3_testf_pred))
	i = i + 1
plt.plot(e3_tot_feat,e3_accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.title('Experiment 3 - Accuracy vs No. of features')
plt.show()

