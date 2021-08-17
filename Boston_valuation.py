from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

#get data

boston_dataset = load_boston()

data = pd.DataFrame(data=boston_dataset.data, 
                    columns = boston_dataset.feature_names)
features = data.drop(['INDUS','AGE'], axis=1)

log_prices = np.log(boston_dataset.target)
target = pd.DataFrame(log_prices, columns=['PRICE'])


#target.shape

CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8

ZILLOW_MEDIAN_PRICE = 583.3
SCALE_FACTOR = ZILLOW_MEDIAN_PRICE / np.median(boston_dataset.target)

property_stats = features.mean().values.reshape(1,11)


regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)

MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)

def get_log_estimate(nr_rooms,
                    students_per_classroom,
                    next_to_river=False,
                    high_confidence=True):
#configure property 
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] =students_per_classroom

    if next_to_river:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0

    #make prediction
    log_estimate = regr.predict(property_stats)[0][0]
 #log_estimate = regr.predict(property_stats)[0][1] access number not array   
    
    #calc range
    if high_confidence:
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
    else: 
        upper_bound = log_estimate + 1*RMSE
        lower_bound = log_estimate - 1*RMSE
        interval = 68
        
    return log_estimate, upper_bound, lower_bound, interval

def get_dollar_estimate(rm, ptratio, chas=False, large_range=True):
    """
    Estimate the price of a property in Boston.
    
    Keywords argument:
    rm -- N. of rooms in the property
    ptratio -- N. of students per teacher in the classom 
    chas -- True: if property is next to river 
    large_range -- True: 95% of conf interval; False: 68% of conf interval
    
    """
    
    if rm < 1 or ptratio <1 :
        print ('not available, please try again')
        return
    

    log_est, upper, lower, conf = get_log_estimate(rm, ptratio,
                                                  next_to_river=chas, high_confidence=large_range)

    dollar_est = np.e**log_est * 1000 * SCALE_FACTOR
    dollar_high = np.e**upper * 1000 * SCALE_FACTOR
    dollar_low = np.e**lower * 1000 * SCALE_FACTOR
    rounded_est = np.round(dollar_est,-3)
    rounded_high = np.round(dollar_high,-3)
    rounded_low = np.round(dollar_low,-3)

    print(f'the estimiate is {rounded_est}')
    print(f'the {conf}% confidence interval valuation range is')
    print(f'between {rounded_low} and {rounded_high}')