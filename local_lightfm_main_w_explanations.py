#%%
from unicodedata import name
import lightfm
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from pandas.core.frame import DataFrame
from local_evaluation import Evaluation
from utils import recommendation_all

from recoxplainer.config import cfg
from recoxplainer.data_reader.data_reader import DataReader
from recoxplainer.explain import ARPostHocExplainer 
from recoxplainer.evaluator import ExplanationEvaluator

#%%
# Lighfm recommendation
train_path = './datasets/w103_train.csv'
evaluation_path = './datasets/w103_test.csv'
w103_df = pd.read_csv(train_path)

#%%
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

lightfm_recommendations = recommendation_all(model, interactions, user_list, user_id_map, item_id_map)

evaluation = Evaluation('', evaluation_path, lightfm_recommendations)
score = evaluation.results()
print(f'Mean Precision: {score}\n')

#%%
# Re-read training set as DataReader
train_path_for_recoxplainer = './datasets/w103_train_for_recoxplainer.csv'
print("reading data...")
tmp_df = pd.read_csv(train_path)
tmp_df = tmp_df.drop('Unnamed: 0', axis=1)
tmp_df = tmp_df.drop('dta_src', axis=1)
tmp_df = tmp_df.drop('deduct_cnt', axis=1)
tmp_df = tmp_df.drop('etl_dt', axis=1)
tmp_df = tmp_df.rename({'cust_no':'userId', 'wm_prod_code':'itemId', 'txn_dt':'timestamp', 'txn_amt':'rating'}, axis=1)
tmp_df = tmp_df.drop_duplicates(subset=['userId', 'itemId'])
tmp_df.to_csv(train_path_for_recoxplainer, index=False)

data = DataReader(filepath_or_buffer= train_path_for_recoxplainer,
                  sep=',',
                  skiprows=1,
                  names=[ 'userId', 'itemId', 'timestamp', 'rating'])

data.make_consecutive_ids_in_dataset()
new_user_id_dict = data.new_user_id.to_dict()['userId']
new_item_id_dict = data.new_item_id.to_dict()['itemId']
original_user_id_dict = data.original_user_id.to_dict()['user_id']
original_item_id_dict = data.original_item_id.to_dict()['item_id']
data.binarize()

#%%
# Transform lightfm recommendation to recoxplainer compatible format

lightfm_recommendations_df = pd.DataFrame.from_dict(dict(lightfm_recommendations), orient='index')
lightfm_recommendations_df.reset_index(level=0, inplace=True)
lightfm_recommendations_df.rename(columns={'index':'userID'}, inplace=True)
lightfm_recommendations_df = lightfm_recommendations_df.melt(id_vars=['userID'], var_name='rank', value_name='itemID')
# lightfm_recommendations_df['rank'] = lightfm_recommendations_df['rank'] + 1

# Map between consecutive and original ID
lightfm_recommendations_df['userID'] = lightfm_recommendations_df['userID'].map(new_user_id_dict)
lightfm_recommendations_df['itemID'] = lightfm_recommendations_df['itemID'].map(new_item_id_dict)

lightfm_recommendations_df = lightfm_recommendations_df.rename({'userID':'userId', 'itemID':'itemId'}, axis=1)
lightfm_recommendations_df = lightfm_recommendations_df[['userId', 'itemId', 'rank']]
# lightfm_recommendations_df = lightfm_recommendations_df.astype('float64')

#%%
# Explain with recoxplainer post-hoc AR
ARexplainer = ARPostHocExplainer(None, lightfm_recommendations_df, data, 
                                    min_support=.001,
                                    max_len=5,
                                    min_threshold=.1,
                                    min_confidence=.1,
                                    min_lift=0.1)
ARexpl = ARexplainer.explain_recommendations()

ex = ExplanationEvaluator(data.num_user)
ex_fi = ex.model_fidelity(ARexpl)

print(ex_fi)
print(ARexpl.head())

#%%
ARexpl_maped = ARexpl.copy(deep=True)
ARexpl_maped['userId'] = ARexpl_maped['userId'].map(original_user_id_dict)
ARexpl_maped['itemId'] = ARexpl_maped['itemId'].map(original_item_id_dict)

for index, row in ARexpl_maped.iterrows():
      row['explanations'] = list(map(lambda x: original_item_id_dict[x], row['explanations']))

print(ARexpl_maped.head())

#%%
# Explain with recoxplainer post-hoc kNN
from recoxplainer.explain import KNNPostHocExplainer
KNNexplainer = KNNPostHocExplainer(None, lightfm_recommendations_df, data)
KNNexpl = KNNexplainer.explain_recommendations()

ex = ExplanationEvaluator(data.num_user)
ex.model_fidelity(KNNexpl)

#%%