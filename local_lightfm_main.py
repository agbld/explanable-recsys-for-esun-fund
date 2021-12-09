#%%
from re import T
from unicodedata import name
import lightfm
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from pandas.core.frame import DataFrame
from local_evaluation import Evaluation
from utils import recommendation_all
from tqdm import tqdm
from datetime import date, datetime, timedelta

train_path = './datasets/w103_train.csv'
evaluation_path = './datasets/w103_test.csv'
train_A_path = './datasets/w103_train_A.csv'
evaluation_A_path = './datasets/w103_test_A.csv'
train_B_path = './datasets/w103_train_B.csv'
evaluation_B_path = './datasets/w103_test_B.csv'

def hybrid_lightfm(threshold = 5, weeks=4, reco_with_pop = False):
      # Build cust_no vs num_of_interactions table(df)
      train_df = pd.read_csv(train_path)
      test_df = pd.read_csv(evaluation_path)
      dataset_df = pd.concat([train_df, test_df], axis=0)
      cust_no_vs_num_of_interactions = dataset_df.drop(dataset_df.columns.difference(['cust_no', 'wm_prod_code']), 1)
      cust_no_vs_num_of_interactions.drop_duplicates(inplace=True)
      cust_no_vs_num_of_interactions.reset_index(inplace=True, drop=True)
      cust_no_vs_num_of_interactions = pd.DataFrame(cust_no_vs_num_of_interactions.groupby('cust_no', as_index=False).size())
      cust_no_vs_num_of_interactions.rename({'size':'num_of_interactions'}, axis=1, inplace=True)

      # Build sparse interactions cust_no list
      sparse_cust_no_list = cust_no_vs_num_of_interactions[cust_no_vs_num_of_interactions['num_of_interactions'] <= threshold]
      sparse_cust_no_list.reset_index(inplace=True, drop=True)
      sparse_cust_no_list = list(sparse_cust_no_list['cust_no'].values)

      # Split dataset into A, B part, and save as csv
      tmp_train = pd.read_csv(train_path)
      tmp_train = tmp_train[~tmp_train['cust_no'].isin(sparse_cust_no_list)]
      tmp_train.to_csv(train_A_path, index=False)
      tmp_test = pd.read_csv(evaluation_path)
      tmp_test = tmp_test[~tmp_test['cust_no'].isin(sparse_cust_no_list)]
      tmp_test.to_csv(evaluation_A_path, index=False)
      tmp_train = pd.read_csv(train_path)
      tmp_train = tmp_train[tmp_train['cust_no'].isin(sparse_cust_no_list)]
      tmp_train.to_csv(train_B_path, index=False)
      tmp_test = pd.read_csv(evaluation_path)
      tmp_test = tmp_test[tmp_test['cust_no'].isin(sparse_cust_no_list)]
      tmp_test.to_csv(evaluation_B_path, index=False)

      # train, recommend A part users with lightfm
      train_A = pd.read_csv(train_A_path)
      if train_A.shape[0]:
            if weeks > 0:
                  train_A['txn_dt'] = pd.to_datetime(train_A['txn_dt'])
                  train_A = train_A[(datetime(2019, 6, 10) - train_A['txn_dt']) < timedelta(weeks=weeks)]

            dataset1 = Dataset()
            dataset1.fit(
                  train_A['cust_no'].unique(), # all the users
                  train_A['wm_prod_code'].unique(), # all the items
            )

            (interactions, weights) = dataset1.build_interactions([(x[1], x[2], x[4]) for x in train_A.values ])
            user_id_map, user_feature_map, item_id_map, item_feature_map = dataset1.mapping()
            user_list = train_A['cust_no'].unique().tolist()

            model = LightFM(loss='warp')
            model.fit(interactions, epochs=10, verbose=True)

            lightfm_recommendations_A = recommendation_all(model, interactions, user_list, user_id_map, item_id_map)

            # A part evaluation
            evaluation = Evaluation('', evaluation_A_path, lightfm_recommendations_A)
            score_A = evaluation.results()
            print(f'[filtered, only A] Mean Precision: {score_A}, threshold = {threshold}\n')
      else:
            lightfm_recommendations_A = {}
            score_A = 0

      # train, recommend B part users with lightfm (or rule-based)
      if len(sparse_cust_no_list):
            # with rule-based
            if reco_with_pop:
                  # calculate top_n, popularity
                  top_n_pop_list = list(train_df.groupby('wm_prod_code', as_index=False).size().sort_values('size', ascending=False)['wm_prod_code'].values[:5])
                  ui_num = train_df.groupby(['cust_no', 'wm_prod_code'], as_index=False).size().sort_values('size', ascending=False)

                  lightfm_recommendations_B = {}
                  with tqdm(total=len(sparse_cust_no_list)) as pbar:
                        for cust_no in sparse_cust_no_list:
                              reco_list = top_n_pop_list.copy()
                              reco_list.reverse()
                              sorted_interact_history = list(ui_num[ui_num['cust_no'] == cust_no]['wm_prod_code'].values)
                              for i in range(5):
                                    if i < len(sorted_interact_history):
                                          reco_list[i] = sorted_interact_history[i]
                                    else: break
                              lightfm_recommendations_B[cust_no] = reco_list
                              pbar.update()
            
                  # B part evaluation
                  evaluation = Evaluation('', evaluation_B_path, lightfm_recommendations_B)
                  score_B = evaluation.results()
                  print(f'[filtered, only B] Mean Precision: {score_B}, threshold = {threshold}\n')
                  
            # with lightfm
            else: 
                  w103_df = pd.read_csv(train_B_path)
                  if weeks > 0:
                        w103_df['txn_dt'] = pd.to_datetime(w103_df['txn_dt'])
                        w103_df = w103_df[(datetime(2019, 6, 10) - w103_df['txn_dt']) < timedelta(weeks=weeks)]

                  dataset1 = Dataset()
                  dataset1.fit(
                        w103_df['cust_no'].unique(), # all the users
                        w103_df['wm_prod_code'].unique(), # all the items
                  )

                  (interactions, weights) = dataset1.build_interactions([(x[1], x[2], x[4]) for x in w103_df.values ])
                  user_id_map, user_feature_map, item_id_map, item_feature_map = dataset1.mapping()
                  user_list = w103_df['cust_no'].unique().tolist()

                  model = LightFM(loss='warp')
                  model.fit(interactions, epochs=10, verbose=True)

                  lightfm_recommendations_B = recommendation_all(model, interactions, user_list, user_id_map, item_id_map)

                  # B part evaluation
                  evaluation = Evaluation('', evaluation_B_path, lightfm_recommendations_B)
                  score_B = evaluation.results()
                  print(f'[filtered, only B] Mean Precision: {score_B}, threshold = {threshold}\n')

      # full evaluation
      lightfm_recommendations = lightfm_recommendations_A
      if len(sparse_cust_no_list):
            lightfm_recommendations.update(lightfm_recommendations_B)
      else:
            score_B = 0
      evaluation = Evaluation('', evaluation_path, lightfm_recommendations_A)
      score = evaluation.results()
      print(f'[filtered, full] Mean Precision: {score}, threshold = {threshold}\n')

      return {'score_A':score_A, 'score_B':score_B, 'score':score, 'threshold':threshold, 'weeks':weeks, 'method':reco_with_pop}

#%%
results = []
for j in [1, 3, 5, 7]:
      for i in [0, 1, 3, 5, 7, 10, 15, 20, 30, 50]:
            print('--------------')
            results.append(hybrid_lightfm(threshold = i, weeks=j * 4, reco_with_pop=False))
            results_df = pd.DataFrame(results)
            results_df.to_csv('results.csv', index=False)
# results.append(hybrid_lightfm(threshold = 1, weeks=4, reco_by_pop=False))

#%%
# print('--------------')
# # without filter
# w103_df = pd.read_csv(train_path)
# w103_df['txn_dt'] = pd.to_datetime(w103_df['txn_dt'])
# w103_df = w103_df[(datetime(2019, 6, 10) - w103_df['txn_dt']) < timedelta(weeks=4)]

# dataset1 = Dataset()
# dataset1.fit(
#       w103_df['cust_no'].unique(), # all the users
#       w103_df['wm_prod_code'].unique(), # all the items
# )

# (interactions, weights) = dataset1.build_interactions([(x[1], x[2], x[4]) for x in w103_df.values ])
# user_id_map, user_feature_map, item_id_map, item_feature_map = dataset1.mapping()
# user_list = w103_df['cust_no'].unique().tolist()

# model = LightFM(loss='warp')
# model.fit(interactions, epochs=10, verbose=True)

# lightfm_recommendations = recommendation_all(model, interactions, user_list, user_id_map, item_id_map)

# # evaluation
# evaluation = Evaluation('', evaluation_path, lightfm_recommendations)
# score = evaluation.results()
# print(f'\n[without filter] Mean Precision: {score}\n')

# results.append({'score_f':score, 'score':score, 'threshold':0, 'method':'trad'})

#%%