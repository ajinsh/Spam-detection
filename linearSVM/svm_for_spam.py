## Author : Ajinkya Shinde
#### Load the required packages
import csv 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
import matplotlib.pyplot as plt


class LinearSVM:

	def __init__(self, csvdata):
		#### Read the dataset and store it into numpy array
		f = open(csvdata,'r')
		csvreader = csv.reader(f)
		data = list(csvreader)

		# Store the data into  numpy array
		spam_arr = np.array(data,float)


		# Seperate the spam dataset into labels and features 
		self.spam_labels = np.asarray(spam_arr[:,-1],dtype='int')
		self.spam_data = np.delete(spam_arr,57, axis = 1)

		# print(self.spam_data.shape)
		# print(self.spam_labels.shape)


	def preprocessing(self, data, labels):	
		### Split data into ~1/2 training and 1/2 test
		self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(self.spam_data, self.spam_labels, test_size = 0.5) 

		#### Scaled training data features

		# train_scaled = preprocessing.scale(train_features)
		# print("train_scaled using skscaling",train_scaled)

		train_features_mean = np.mean(self.train_features,axis = 0)
		train_features_std = np.std(self.train_features,axis = 0)


		for each in self.train_features:
			for i in range(57):
				if train_features_std[i] != 0:
					each[i] = (each[i]-train_features_mean[i])/train_features_std[i]
				else:
					each[i] = 0
		# print("train_features without using skscaling",train_features)	

		#### Scaled testing data features using train data mean and train data std
		for each in self.test_features:
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

	def utility_print(self):
		print(str(self.spam_data[:10]))
		# print(str(self.train_features[:10]))

	def plot_data(self,x, y, xlabname, ylabname, title):
		plt.plot(x,y)
		plt.ylabel(ylabname)
		plt.xlabel(xlabname)
		plt.title(title)
		plt.show()

	def experiment1_analysis(self):	
		# ##################################################################
		# ################# Experiment 1  #########################
		# ##################################################################

		# self.model = SVC(kernel='linear')
		# self.model.fit(self.train_features, self.train_labels)  
		# self.test_pred = self.model.predict(self.test_features)


		# # print(classification_report(test_labels,test_pred))  
		# # print(confusion_matrix(test_labels,test_pred))  

		# print(test_labels.shape[0])
		# print(test_pred.shape[0])

		TP = 0
		FP = 0
		TN = 0
		FN = 0


		for i in range(self.test_labels.shape[0]):
			if self.test_labels[i] == 0:
				if self.test_labels[i] == self.test_pred[i]:
					TP = TP + 1
				else:
					FN = FN + 1
			else:
				if self.test_labels[i] == self.test_pred[i]:
					TN = TN + 1
				else:
					FP = FP + 1

		print("Recall for non-spam emails    :",round(TP/(TP + FN),2))
		print("Precision for non-spam emails :",round(TP/(TP + FP),2))
		print("Recall for spam emails        :",round(TN/(TN + FP),2))
		print("Precision for spam emails     :",round(TN/(TN + FN),2))
		print("Accuracy:",accuracy_score(self.test_labels,self.test_pred))

		fpr, tpr, thresholds = roc_curve(self.test_labels,self.model.decision_function(self.test_features))
		# print('fpr',fpr.shape)
		# print('tpr',tpr.shape)
		# print('thresholds', thresholds.shape)
		self.plot_data(fpr,tpr,'True Positive Rate','False Positive Rate','Receiver Operating Characterestic (ROC) Curve')

	def experiment2_analysis(self):
		# ##################################################################
		# ################# Experiment 2 ###################################
		# ##################################################################
		# Co-effient for support vectors - α   
		alpha_support_vec = self.model.dual_coef_


		# Support vectors returned by the model
		support_vec = self.model.support_vectors_

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
		self.model1 = SVC(kernel='linear')
		e2_accuracy = []
		e2_tot_feat = []
		for i in range(57):
			e2_trainf = self.train_features[:,idx_ftselec[0:i+1]]
			e2_testf = self.test_features[:,idx_ftselec[0:i+1]]
			if i == 0:
				feat_select = 2
			else:
				feat_select = feat_select + 1
			e2_tot_feat.append(feat_select)
			self.model1.fit(e2_trainf, self.train_labels)
			e2_testf_pred  = self.model1.predict(e2_testf)
			e2_accuracy.append(accuracy_score(self.test_labels,e2_testf_pred))
			i = i + 1


		# Plot the accuracy vs m (no. of features)
		self.plot_data(e2_tot_feat,e2_accuracy,'Number of features','Accuracy','Experiment 2 - Accuracy vs No. of features')
		# plt.ylabel('Accuracy')
		# plt.xlabel('Number of features')
		# plt.title('Experiment 2 - Accuracy vs No. of features')
		# plt.show()
	
	def experiment3_analysis(self):
		# ##################################################################
		# ################# Experiment 3 ###################################
		# ##################################################################

		self.model2 = SVC(kernel='linear')
		e3_accuracy = []
		e3_tot_feat = []


		for i in range(57):
			rand_idx = np.random.choice(np.arange(57),i+1)
			e3_trainf = self.train_features[:,rand_idx]
			e3_testf = self.test_features[:,rand_idx]
			if i == 0:
				feat_select1 = 2
			else:
				feat_select1 = feat_select1 + 1
			e3_tot_feat.append(feat_select1)
			self.model2.fit(e3_trainf, self.train_labels)
			e3_testf_pred  = self.model2.predict(e3_testf)
			e3_accuracy.append(accuracy_score(self.test_labels,e3_testf_pred))
			i = i + 1
		self.plot_data(e3_tot_feat,e3_accuracy,'Accuracy','Number of features','Experiment 3 - Accuracy vs No. of features')

		
lSVM = LinearSVM('spambase.data')
lSVM.preprocessing(lSVM.spam_data, lSVM.spam_labels) 
# lSVM.utility_print()
lSVM.model =  SVC(kernel='linear')
lSVM.model.fit(lSVM.train_features, lSVM.train_labels)  
lSVM.test_pred = lSVM.model.predict(lSVM.test_features)
lSVM.experiment1_analysis()
# lSVM.experiment2_analysis()
# lSVM.experiment3_analysis()