import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression,LinearRegression
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class MyLogisticRegression:
    def __init__(self, dataset_num, perform_test):
        self.training_set = None
        self.test_set = None
        self.model_logistic = None
        self.model_linear = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.perform_test = perform_test
        self.dataset_num = dataset_num
        self.read_csv(self.dataset_num)

    def read_csv(self, dataset_num):
        if dataset_num == '1':
            train_dataset_file = 'train_q1_1.csv'
            test_dataset_file = 'test_q1_1.csv'
        elif dataset_num == '2':
            train_dataset_file = 'train_q1_2.csv'
            test_dataset_file = 'test_q1_2.csv'
        else:
            print("unsupported dataset number")
            
        self.training_set = pd.read_csv(train_dataset_file, sep=',', header=0)
        self.X_train = self.training_set[['exam_score_1', 'exam_score_2']].values
        self.y_train = self.training_set['label'].values

        if self.perform_test:
            self.test_set = pd.read_csv(test_dataset_file, sep=',', header=0)
            self.X_test = self.test_set[['exam_score_1', 'exam_score_2']].values
            self.y_test = self.test_set['label'].values

        
        
    def model_fit_linear(self):
        '''
        initialize self.model_linear here and call the fit function
        '''
        self.model_linear = LinearRegression()
        self.model_linear.fit(self.X_train, self.y_train)
    
    def model_fit_logistic(self):
        '''
        initialize self.model_logistic here and call the fit function
        '''
        self.model_logistic = LogisticRegression()
        self.model_logistic.fit(self.X_train, self.y_train)
    
    def model_predict_linear(self):
        '''
        Calculate and return the accuracy, precision, recall, f1, support of the model.
        '''
        self.model_fit_linear()
        accuracy = 0.0
        precision, recall, f1, support = np.array([0,0]), np.array([0,0]), np.array([0,0]), np.array([0,0])
        assert self.model_linear is not None, "Initialize the model, i.e. instantiate the variable self.model_linear in model_fit_linear method"
        assert self.training_set is not None, "self.read_csv function isn't called or the self.trianing_set hasn't been initialized "
        
        if self.X_test is not None:
            # perform prediction here
            y_pred = self.model_linear.predict(self.X_test)
            y_pred = np.where(y_pred >= 0.5, 1, 0)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(self.y_test, y_pred)
            print(accuracy)
        
        assert precision.shape == recall.shape == f1.shape == support.shape == (2,), "precision, recall, f1, support should be an array of shape (2,)"
        return [accuracy, precision, recall, f1, support]

    def model_predict_logistic(self):
        '''
        Calculate and return the accuracy, precision, recall, f1, support of the model.
        '''
        self.model_fit_logistic()
        accuracy = 0.0
        precision, recall, f1, support = np.array([0,0]), np.array([0,0]), np.array([0,0]), np.array([0,0])
        assert self.model_logistic is not None, "Initialize the model, i.e. instantiate the variable self.model_logistic in model_fit_logistic method"
        assert self.training_set is not None, "self.read_csv function isn't called or the self.trianing_set hasn't been initialized "
        if self.X_test is not None:
            y_pred = self.model_logistic.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            print(accuracy)
            precision, recall, f1, support = precision_recall_fscore_support(self.y_test, y_pred)
            pass
        assert precision.shape == recall.shape == f1.shape == support.shape == (2,), "precision, recall, f1, support should be an array of shape (2,)"
        return [accuracy, precision, recall, f1, support]
    
    def modelPlot(self):
        x_min, x_max = self.X_test[:, 0].min() - 1, self.X_test[:, 0].max() + 1
        y_min, y_max = self.X_test[:, 1].min() - 1, self.X_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Predict using the linear model for the entire grid
        Z1 = self.model_linear.predict(np.c_[xx.ravel(), yy.ravel()])
        Z1 = Z1.reshape(xx.shape)

        # Plot the decision boundary for linear regression
        axs[0].contourf(xx, yy, Z1, alpha=0.8, cmap=plt.cm.coolwarm)
        axs[0].scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, edgecolors='k', marker='o', s=20, cmap=plt.cm.coolwarm)
        axs[0].set_title("Decision Boundary for Linear Regression")
        axs[0].set_xlabel("Exam Score 1")
        axs[0].set_ylabel("Exam Score 2")

        # Predict using the logistic model for the entire grid
        Z2 = self.model_logistic.predict(np.c_[xx.ravel(), yy.ravel()])
        Z2 = Z2.reshape(xx.shape)

        # Plot the decision boundary for logistic regression
        axs[1].contourf(xx, yy, Z2, alpha=0.8, cmap=plt.cm.coolwarm)
        axs[1].scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, edgecolors='k', marker='o', s=20, cmap=plt.cm.coolwarm)
        axs[1].set_title("Decision Boundary for Logistic Regression")
        axs[1].set_xlabel("Exam Score 1")
        axs[1].set_ylabel("Exam Score 2")

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Regression')
    parser.add_argument('-d','--dataset_num', type=str, default = "1", choices=["1","2"], help='string indicating datset number. For example, 1 or 2')
    parser.add_argument('-t','--perform_test', action='store_true', help='boolean to indicate inference')
    args = parser.parse_args()
    classifier = MyLogisticRegression(args.dataset_num, args.perform_test)

    # command prompt must set perorm_test in order for these to display anything besides 0
    acc = classifier.model_predict_linear()
    print("Accuracy: {:.4f}".format(acc[0]))
    print("Linear class 0 | p r f1 sup = {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(acc[1][0], acc[2][0], acc[3][0], acc[4][0]))
    print("Linear class 1 | p r f1 sup = {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(acc[1][1], acc[2][1], acc[3][1], acc[4][1]))
    print("------")
    
    acc = classifier.model_predict_logistic()
    print("Accuracy: {:.4f}".format(acc[0]))
    print("Logistic class 0 | p r f1 sup = {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(acc[1][0], acc[2][0], acc[3][0], acc[4][0]))
    print("Logistic class 1 | p r f1 sup = {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(acc[1][1], acc[2][1], acc[3][1], acc[4][1]))
    classifier.modelPlot()
    
    