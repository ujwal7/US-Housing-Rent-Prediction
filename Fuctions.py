import pandas as pd
import numpy as np
from time import time
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,f1_score
from pretty_confusion_matrix import pp_matrix
import seaborn as sns

# This function is used to impute missing values of the categorical variables with the mode value of the given column
def impute_nan_with_mode(DataFrame,column):
    """
    Takes in a Dataframe and columns name and returns the Dataframe with a new column with the imputed values based on mode of column.
    :param DataFrame: Dataframe to be parsed
    :param column: Name of the column which has to be imputed
    :return: Dataframe consisting the new imputed column
    """
    most_frequent_category=DataFrame[column].mode()[0]
    DataFrame[column + "_Imputed"] = DataFrame[column]
    DataFrame[column + "_Imputed"].fillna(most_frequent_category,inplace=True)


# This function is used to calculate the IQR, lower bound and upper bound value for a given column
def calcuateIQR(DataFrame,column):
    """
    Takes in a Dataframe and columns name and returns the Dataframe with a new column with the imputed values based on mode of column.
    :param DataFrame: Dataframe to be parsed
    :param column: Name of the column which has to be imputed
    :return: Prints the IQR, Lower Bound, Upper Bound values and the total number of Outliers
    """
    global lower,upper
    q25, q75 = np.quantile(DataFrame[column], 0.25), np.quantile(DataFrame[column], 0.75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    print('The IQR is',iqr)
    print('The lower bound value is', lower)
    print('The upper bound value is', upper)
    df1 = DataFrame[DataFrame[column] > upper]
    df2 = DataFrame[DataFrame[column] < lower]
    return print('Total number of outliers are', df1.shape[0]+ df2.shape[0])


# This function finds the labels which are shared just by 1% of the listings in the dataset
def findRareLabels(df, var, rare_perc):
    """
    Takes in a Dataframe and columns name and percentage
    :param DataFrame: Dataframe to be parsed
    :param var: Name of the column which observed
    :param rare_perc: Defines how many percentage of the category to be observed in our case it will be 1% i.e 0.01
    :return: Percentage of ibservations per category
    """
    tmp = df.groupby(var)['price'].count() / len(df)
    return tmp[tmp < rare_perc]


# This function finds the labels which are shared by more than a certain % of the listings in the data set.
def findFrequentLabels(df, var, rare_perc):
    """
    Takes in a Dataframe and columns name and percentage
    :param DataFrame: Dataframe to be parsed
    :param var: Name of the column which observed
    :param rare_perc: Defines how many percentage of the category to be observed in our case it will be 1% i.e 0.01
    :return: Percentage of observations per category
    """
    tmp = df.groupby(var)['price'].count() / len(df)
    return tmp[tmp > rare_perc].index.values


# This function will assign discrete values to the strings of the variables,
# so that the smaller value corresponds to the category that shows the smaller mean house sale price.
# The main purpose of this function is encoding the values.
def repalceCategoricalWithNumerical(data, var, target):
    """
    Takes in a Dataframe and columns name and percentage
    :param DataFrame: Dataframe to be parsed
    :param var: Name of the column which observed
    :param target: Value according to which the mean is to be calculated
    :return: Returns the values to replace the categorical strings with integers
    """
    ordered_labels = data.groupby([var])[target].mean().sort_values().index
    ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}
    data[var] = data[var].map(ordinal_label)


#This function will call used for Classification. It will predict and then print the feature importance as well    
def predict_model(model, X_train, X_test, y_train, y_test):
    """
    Takes in the model object in which we have to make the prediction and the the Training, Test Dataset split values.
    :param model: Model to be used for prediction
    :param X_train: Features of Traning Data set
    :param X_test: Features of Test Data set
    :param y_train: Target value of Traning Dataset
    :param y_test: Target value of Test
    """
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    accTest=accuracy_score(y_test, predictions)
    a = confusion_matrix(y_test,predictions)
    conf_df = pd.DataFrame(a)
    cmap = 'OrRd'
    pp_matrix(conf_df, cmap=cmap)
    print('Feature Importance for Random Forest Model while predicting Region')
    importance = model.feature_importances_
    feature_imp = pd.Series(model.feature_importances_,index=X_train.columns).sort_values(ascending=False)
    plt.rcParams["figure.figsize"] = (12,20)
    sns.barplot(x=feature_imp,y=X_train.columns,palette='Reds_r')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title('Visualizing Important Features')
    plt.legend()
    plt.show()    
    


# This is the main fucntion of the project which makes the prediction and plots the plot with respect to Actual and Predicted Rent.
def predictRegressor(model, X_train, X_test, y_train, y_test, X2_train, y2_train, X2_test, y2_test):
    """
    Takes in the model object in which we have to make the prediction and the the Training, Test Dataset split values.
    :param model: Model to be used for prediction
    :param X_train: Features of Traning Data set
    :param X_test: Features of Test Data set
    :param y_train: Target value of Traning Dataset
    :param y_test: Target value of Test
    :param X2_train: Features of Traning Data set (Scaled)
    :param X2_test: Features of Test Data set (Scaled)
    :param y2_train: Target value of Traning Dataset (Scaled)
    :param y2_test: Target value of Test Dataset (Scaled)   
    :return: Prints the MAE, RMSE, R2_Score for the model provided. Also prints the plot of Actual vs Pridcited Rent
    """
    # Start the clock, train the regressor, then stop the clock
    print("\n")
    start = time()
    model.fit(X_train, y_train)
    end = time()
    print("Trained model without scaling in {:.4f} seconds".format(end - start))
    start = time()
    rf1_pred = model.predict(X_test)
    end = time()
    print("Made predictions for without scaling dataset in {:.4f} seconds.".format(end - start))
    print('\nAll features, No scaling:')
    print('MAE:', metrics.mean_absolute_error(y_test, rf1_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rf1_pred)))
    print('R2_Score: ', metrics.r2_score(y_test, rf1_pred))
    start = time()
    model.fit(X2_train, y2_train)
    end = time()
    print("Trained model with scaling in {:.4f} seconds".format(end - start))
    start = time()
    rf2_pred = model.predict(X2_test)
    end = time()
    print("Made predictions for with scaling dataset in {:.4f} seconds.".format(end - start))
    print('\nAll features, with scaling:')
    print('MAE:', metrics.mean_absolute_error(y2_test, rf2_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y2_test, rf2_pred)))
    print('R2_Score: ', metrics.r2_score(y2_test, rf2_pred))

    #Plotting the performace chart
    pred = pd.DataFrame(data={'Predicted_Rent': np.exp(rf1_pred), 'Actual_Rent': np.exp(y_test)})
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    ax1.scatter(pred['Actual_Rent'], pred['Predicted_Rent'], color='tab:cyan', marker=".", label='Predicted', alpha=0.5)
    ax1.scatter(pred['Actual_Rent'], pred['Actual_Rent'], s=10, color='tab:orange', marker="s", label='Actual')
    plt.xlabel('Actual Rent')
    plt.ylabel('Predicted Rent')
    plt.legend(['R2_Score= {:.4f}\nRMSE= {:.4f} '.format(metrics.r2_score(y_test, rf1_pred), np.sqrt(metrics.mean_squared_error(y_test, rf1_pred)))],loc='best')
    plt.title(model)
    plt.grid()
    plt.show()
    
    
# These functions are used to change categorical variables to numerical variables.

# Encode categorical variable for laundry options.
def encode_laundry(laundry):
    """
    Takes in a column of a Dataframe and returns 0 or 1 if 'no laundy on site' is the value, else 1 .
    :param laundry: Dataframe Column to be parsed (laundry_options)
    :return: 0 if no laundry on site, else 1
    """
    if laundry == 'no laundry on site':
        return 0
    else:
        return 1
    
# Encode categorical variable for garage options    
def encode_garage(garage):
    """
    Takes in a column of a Dataframe and returns 0 or 1 if a given value is present in the given column or not.
    :param DataFrame: Dataframe Column to be parsed (parking_options)
    :return: 1 either of detached garage','attached garage','carport','valet parking' is there, else 0
    """
    if garage in ['detached garage','attached garage','carport','valet parking']:
        return 1
    else:
        return 0

# Encode categorical variable house_type    
def encode_type(house_type):
    """
    Takes in a column of a Dataframe and returns 0 or 1 if a particular type of house is there or not
    :param DataFrame: Dataframe Column to be parsed (type)
    :return: 1 if house_type is in apartment, condo or duplex 
    """
    if house_type in ['apartment']:
        return 1
    else:
        return 0
    
# Encode categorical variable region      
def encode_region(region):
    """
    Takes in a column of a Dataframe and returns 0,1,2,3,4 based on the region
    :param DataFrame: Dataframe Column to be parsed (region)
    :return: 0 if region is Boston, 1 if it is new york city, 2 if it is los angeles, 3 if it is chicago and 4 if it is atlanta  
    """
    if region == 'boston':
        return 0
    if region == 'new york city':
        return 1
    if region == 'los angeles':
        return 2
    if region == 'chicago':
        return 3
    if region == 'atlanta':
        return 4

"""
FUNCTIONS ADDED BELOW ARE USED TO CREATE NEW FEATURES IN OUR DATASET
IF A CERTAIN SUBSTRING IS PRESENT IN THE PARAMETER OF THE FUCNTION THEN THESE FUNCTIONS RETURN TRUE, ELSE FALSE
"""

def has_pool(description):
    """
    Takes in a column of a Dataframe and returns 0 or 1 if pool is present in the value of the given column or not.
    :param DataFrame: Dataframe Column to be parsed
    :return: 1 is pool is there, 0 if its not there
    """
    if 'pool' in description.lower() or 'swimming' in description.lower():
        return 1
    else:
        return 0


def has_grill(description):
    """
    Takes in a column of a Dataframe and returns 0 or 1 if grill is present in the value of the given column or not.
    :param DataFrame: Dataframe Column to be parsed
    :return: 1 is grill is there, 0 if its not there
    """
    if 'grill' in description.lower() or 'grilling' in description.lower():
        return 1
    else:
        return 0


def has_fireplace(description):
    """
    Takes in a column of a Dataframe and returns 0 or 1 if fireplace is present in the value of the given column or not.
    :param DataFrame: Dataframe Column to be parsed
    :return: 1 is fireplace is there, 0 if its not there
    """
    if 'fireplace' in description.lower() or 'fire pits' in description.lower():
        return 1
    else:
        return 0


def has_gymNearBy(description):
    """
    Takes in a column of a Dataframe and returns 0 or 1 if gym is present in the value of the given column or not.
    :param DataFrame: Dataframe Column to be parsed
    :return: 1 is gym is there, 0 if its not there
    """
    if 'gym' in description.lower() or 'fitness' in description.lower():
        return 1
    else:
        return 0


def schoolCollegeNearBy(description):
    """
    Takes in a column of a Dataframe and returns 0 or 1 if school or college is present in the value of the given column or not.
    :param DataFrame: Dataframe Column to be parsed
    :return: 1 is school or college is there, 0 if its not there
    """
    if 'school' in description.lower() or 'college' in description.lower():
        return 1
    else:
        return 0


def wifiFacilities(description):
    """
    Takes in a column of a Dataframe and returns 0 or 1 if Wifi is present in the value of the given column or not.
    :param DataFrame: Dataframe Column to be parsed
    :return: 1 is wifi or wi-fi is there, 0 if its not there
    """
    if 'wifi' in description.lower() or 'wi-fi' in description.lower():
        return 1
    else:
        return 0


def valetService(description):
    """
    Takes in a column of a Dataframe and returns 0 or 1 if valet Service is present in the value of the given column or not.
    :param DataFrame: Dataframe Column to be parsed
    :return: 1 is valet service is there, 0 if its not there
    """
    if 'valet' in description.lower():
        return 1
    else:
        return 0


def shoppingNearBy(description):
    """
    Takes in a column of a Dataframe and returns 0 or 1 if shopping complex is present in the value of the given column or not.
    :param DataFrame: Dataframe Column to be parsed
    :return: 1 is shopping complex is there, 0 if its not there
    """
    if 'shopping' in description.lower():
        return 1
    else:
        return 0


def sportsPlaygroundNearBy(description):
    """
    Takes in a column of a Dataframe and returns 0 or 1 if ground is present in the value of the given column or not.
    :param DataFrame: Dataframe Column to be parsed
    :return: 1 is sports or playground or tennis or soccer is there, 0 if its not there
    """
    if 'sport' in description.lower() or 'sports' in description.lower() or 'tennis' in description.lower() or 'soccer' in description.lower():
        return 1
    else:
        return 0


def diningNearBy(description):
    """
    Takes in a column of a Dataframe and returns 0 or 1 if dining area is present in the value of the given column or not.
    :param DataFrame: Dataframe Column to be parsed
    :return: 1 is dining is there, 0 if its not there
    """
    if 'dining' in description.lower():
        return 1
    else:
        return 0
