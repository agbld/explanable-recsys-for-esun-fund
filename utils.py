import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def top5_recommendation_user(model, interactions, user_id, user_dict, 
                               item_dict,threshold = 0,nrec_items = 5):
    
    n_users, n_items = interactions.shape
    user_x = user_dict[user_id]
    scores = pd.Series(model.predict(user_x,np.arange(n_items)))
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    
    #known_items = list(pd.Series(interactions.loc[user_id,:] \
    #                             [interactions.loc[user_id,:] > threshold].index).sort_values(ascending=False))
    
    #scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]
    pred = [k for k, v in item_dict.items() if v in return_score_list]
    
    return user_id, pred

def recommendation_all(model, intersections, user_li, user_dict, item_dict):

    predictions = defaultdict(list)

    for u in tqdm(user_li, total=len(user_li)):
        user_id, pred = top5_recommendation_user(model, intersections, u, user_dict, item_dict)
        predictions[user_id] = pred

    return predictions
