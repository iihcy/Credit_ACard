# coding:utf-8
# @Author:iihcy
import xgboost as xgb
from sklearn.grid_search import GridSearchCV

# 网格搜索策略--XGBoost
def get_params_init():
    params_init = {
        'base_score':0.5,
        'booster':'gbtree',
        'gamma':0,
        'learning_rate':0.1,
        'max_depth':8,
        'n_estimators':10,
        'objective':'binary:logistic',
        'random_state':0,
        'reg_lambda':1,
        'subsample':0.7,
    }
    return params_init

def get_tree_params():
    params = {
        'n_estimators':[10, 50, 100, 150, 200],
    }
    return params

def get_subsample_params():
    params = {
        'subsample':[0.5,0.6,0.7,0.8],
    }
    return params

def get_learn_params():
    params = {
        'learning_rate':[0.01,0.05,0.1,0.15],
    }
    return params

def get_gr_params():
    params = {
        'gamma':[0.1,0.5,1],
        'reg_lambda':[0.1,0.5,1,2],
    }
    return params

def grid_params_model(model, df, label):
    # 参数组合
    grid_params = [get_tree_params(), get_subsample_params(), get_learn_params(), get_gr_params()]
    for _params_ in grid_params:
        gcv = GridSearchCV(estimator=model, param_grid=_params_, cv=5, iid=False)
        gcv.fit(df, label)
        print(gcv.scorer_, gcv.best_params_, gcv.best_score_)
        # 参数更新
        gcv_params =  model.get_params()
        gcv_params.update(gcv.best_params_)
        model.set_params(**gcv_params)
        print('the other params!')
    return model

def xgb_model_run(df, label):
    params_init = get_params_init()
    # 初始模型
    model_xgb = xgb.XGBClassifier(**params_init)
    print(model_xgb.get_params())
    model_xgb = grid_params_model(model_xgb, df, label)
    return model_xgb