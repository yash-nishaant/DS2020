#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pickle

# Custom scaler class

class CustomScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy, with_mean, with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

# Special class for the model
class absenteeism_model():
    
    # Initialization
    def __init__(self, model_file, scaler_file):
        # Read 'model' and 'scaler' files
        with open('model','rb') as model_file, open('scaler', 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
        
    # Function for preprocessing data as described in the Absenteeism Model (Preprocessing) notebook
    def load_and_clean_data(self, data_file):
        # Load csv file
        df = pd.read_csv(data_file,delimiter=',')
        # Make a copy for dataframe with predictions
        self.df_with_predictions = df.copy()
        # Drop the 'ID' column
        df = df.drop(['ID'], axis=1)
        # To preserve code created in Absenteeism Model notebook, add column of 'NaN' strings for Absenteeism Time in Hours
        df['Absenteeism Time in Hours']='NaN'
        
        # Create dataframe containing dummy_variables for ALL reasons of absence
        reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)
        
        # Split reason_columns into 4 types
        reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
        reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
        reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
        reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)
        
        # Avoid multicollinearity, drop 'Reason for Absence' column
        df = df.drop(['Reason for Absence'], axis=1)
        
        # Concatenate df and reason groups
        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)
        
        # Assign names to reason columns and rearrange them in dataframe
        column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                       'Daily Work Load Average', 'Body Mass Index', 'Education',
                       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
        df.columns = column_names
        
        column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense', 
                                  'Distance to Work', 'Age','Daily Work Load Average', 'Body Mass Index', 'Education',
                                   'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_reordered]
        
        # Convert 'Date' to datetime from string
        df['Date'] = pd.to_datetime(df['Date'], format = '%d/%m/%Y')
        
        # Extract months from date and add to dataframe
        list_months = []
        for i in range(df.shape[0]):
            list_months.append(df['Date'][i].month)
        
        df['Month'] = list_months
        
        # Extract day of the week from date
        df['Day'] = df['Date'].apply(lambda x: x.weekday())
        
        # Drop 'Date' and rearrange columns
        df = df.drop(['Date'], axis=1)
        
        column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month', 'Day', 'Transportation Expense', 
                                  'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education',
                                  'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_reordered]
        
        # Map and categorize 'Education' feature
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})
        
        # Replace NaN values
        df = df.fillna(value=0)
        
        # Drop the original absenteeism time column
        df = df.drop(['Absenteeism Time in Hours'], axis=1)
        
        # Drop the variables we concluded were insignificant
        df = df.drop(['Day', 'Daily Work Load Average', 'Distance to Work'], axis=1)
        
        # Call the preprocessed data
        self.preprocessed_data = df.copy()
        
        # Scaling the df
        self.data = self.scaler.transform(df)
        
    # Function for outputting the probability of an observation being 1
    def predicted_probability(self):
        if (self.data is not None):
            pred = self.reg.predict_proba(self.data)[:,1]
            return pred
    
    # Function which outputs category information (0 or 1)
    def predicted_output_category(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs
    
    # predict the outputs and probabilities and add columns with these values to new dataframe
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data        

