#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##Join three tables and extract statistics
import pandas as pd
import numpy as np

#import data
application = pd.read_csv('../data/application_train.csv')
bureau = pd.read_csv('../data/bureau.csv')
previous = pd.read_csv('../data/previous_application.csv')

#get the following statistics from bureau.csv: number of CB reported credits
#number of total days past due(CB),exist of bad debt flag
#nearest application day before current application(CB)
#longest remaining duration of CB credit,maximal amount overdue for all credits(CB),
#total amount overdue for all credits(CB),number of credit prolonged(CB)
CB = bureau.groupby(['SK_ID_CURR'], as_index=False).agg({
    'CREDIT_ACTIVE':[len,lambda x:sum(x=='Active'),lambda x:int(any(x=='Bad debt'))],
    'DAYS_CREDIT':[max],
    'DAYS_CREDIT_ENDDATE':[max],
    'AMT_CREDIT_MAX_OVERDUE':[max,np.sum],
    'CNT_CREDIT_PROLONG':[np.sum]
    })
CB.columns = CB.columns.to_flat_index()
CB.columns = ['SK_ID_CURR','CB_NUMBER_CREDITS','CB_NUMBER_ACTIVE_CREDITS','CB_BAD_DEBT'
               ,'CB_NEAREST_DAYS_CREDIT','CB_LONGEST_REMAIN','CB_MAX_AMT_OVERDUE',
             'CB_TTL_AMT_OVERDUE','CB_TTL_CREDIT_PROLONG']

#get the following statistics from previous_application.csv:
#number of previous applications(HC),number of previous applications approved(HC)
PA = previous.groupby(['SK_ID_CURR'], as_index=False).agg({
    'FLAG_LAST_APPL_PER_CONTRACT':[lambda x:sum(x=='Y')],
    'NAME_CONTRACT_STATUS':[lambda x:sum(x=='Approved')]
    })
PA.columns = PA.columns.to_flat_index()
PA.columns = ['SK_ID_CURR','HC_NUMBER_APPLICATION','HC_NUMBER_APPROVED']


#join the result 
data1=application.join(CB.set_index('SK_ID_CURR'), on='SK_ID_CURR')
data2 = data1.join(PA.set_index('SK_ID_CURR'), on='SK_ID_CURR')

#drop SK_ID_CURR
data = data2.drop(['SK_ID_CURR'],axis=1)

#save in csv
data.to_csv(r'../data/data.csv', index = False, header=True)

