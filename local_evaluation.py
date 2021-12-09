import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

class Evaluation:
    
    def __init__(self, date, path, pred):
        self.today = date
        self.path = path
        self.pred = pred
        self.ans = self.answer(self.path)
    
    def show(self):
        print(f"Date: {self.today}\n") 
        coverage = len(set(self.pred.keys()) & set(self.ans.keys()))       
        print(f"Uppper-Bound: {coverage}\n")
 
    def answer(self, path):
        df = self.read(path)
        return df.groupby('cust_no')['wm_prod_code'].apply(list).to_dict()

    def read(self, path):
        return pd.read_csv(path, usecols=['cust_no', 'wm_prod_code'])  

    def results(self):
        p = 0
        count = len(self.pred)
        for u, pred in tqdm(self.pred.items(), total=count):
            p += self.precision_at_5(u, pred)
        return p/count         
    
    def precision_at_5(self, user, pred):
        try:
            y_true = self.ans[user]
            tp = len(set(y_true) & set(pred))
            return tp/5
        except:
            return 0
