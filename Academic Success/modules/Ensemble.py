# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 23:43:21 2024

@author: Stephenson
"""
import os
import pandas as pd

import_path = r"C:\Users\{}\Documents\GitHub\Projects\Academic Success".format(os.getlogin())

def most_frequent_value(row):
    return row.mode().iloc[0]


ann_result = pd.read_csv(import_path + r'\outputs\submission.csv')
cat_result = pd.read_csv(import_path + r'\outputs\submission_cat.csv')
lgbm_result = pd.read_csv(import_path + r'\outputs\submission_lgbm.csv')

submission = pd.merge(ann_result, cat_result, on = 'id', suffixes=('_ann', '_cat'))
submission = pd.merge(submission, lgbm_result, on = 'id')
submission.rename(columns = {'Target':'Target_lgbm'}, inplace=True)

output_id = submission['id']
submission.drop('id', axis = 'columns', inplace=True)

submission['Target'] = submission.apply(most_frequent_value, axis = 1)

submission = pd.concat([output_id, submission['Target']], axis = 1)
submission.to_csv(import_path + r'\outputs\submission_ensembv3.csv', index=False)
