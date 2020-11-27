import pandas as pd
import numpy as np

#read data
data = pd.read_csv('../data/data_join.csv')

# Remove the first two variables by the number of missing value
data = pd.DataFrame.dropna(data, axis=0,subset=['CB_MAX_AMT_OVERDUE','COMMONAREA_MEDI'])

#add indicator whether the customer has records in Credit Bureau. CB_FLAG=1 no record in CB
data['CB_FLAG']=pd.DataFrame(data.iloc[:,-10:-2].isna().all(axis=1).astype(int))
#For all customer without CB records, imputation with zero
mask1 = data.iloc[:,-10:-2].isna().all(axis=1)
mask2 = data.columns[-10:-2]
data.loc[mask1,mask2] = data.loc[mask1,mask2].fillna(0)

#add indicator whether the customer has records in Credit Bureau. CB_FLAG=1 no record in CB
data['HC_NEW_FLAG']=pd.DataFrame(data.iloc[:,-2:].isna().all(axis=1).astype(int))
#For all customer without CB records, imputation with zero
mask1 = data.iloc[:,-2:].isna().all(axis=1)
mask2 = data.columns[-2:]
data.loc[mask1,mask2] = data.loc[mask1,mask2].fillna(0)

# replace missing value with median for numeric
N_colunms= list(data.select_dtypes(exclude=['object']).columns)
data[N_colunms] = data[N_colunms].fillna(data[N_colunms].median())
# replace missing value with 'unknown' for object
O_colunms= list(data.select_dtypes(include=['object']).columns)
data[O_colunms] = data[O_colunms].fillna('unknown')


#save data
data.to_csv(r'../data/data_clean.csv', index = False, header=True)


