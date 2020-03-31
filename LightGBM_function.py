# coding:utf-8
# @Author:iihcy
import lightgbm as lgb
from sklearn.grid_search import GridSearchCV

# 网格搜索策略--lightGBM
def get_params_init():
    params_init = {
        'boosting_type': 'gbdt',
        'n_estimators': 100,
        'learning_rate': 0.01,
        'max_depth': 5,
        'subsample': 0.7,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
    }
    return params_init

def get_tree_params():
    params = {
        'n_estimators': [10, 50, 100, 150, 200],
        'max_depth': [5, 6, 7, 8],
    }
    return params

def get_subsample_params():
    params = {
        'subsample': [0.5, 0.6, 0.7, 0.8],
    }
    return params

def get_learn_params():
    params = {
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
    }
    return params

def get_al_params():
    params = {
        'reg_alpha': [0.1, 0.5, 1, 2],
        'reg_lambda': [0.1, 0.5, 1, 2],
    }
    return params

def grid_params_model(model, df, label):
    # 参数组合
    grid_params = [get_tree_params(), get_subsample_params(), get_learn_params(), get_al_params()]
    for _params_ in grid_params:
        gcv = GridSearchCV(estimator=model, param_grid=_params_, cv=5, iid=False)
        gcv.fit(df, label)
        print(gcv.scorer_, gcv.best_params_, gcv.best_score_)
        # 参数更新
        gcv_params = model.get_params()
        gcv_params.update(gcv.best_params_)
        model.set_params(**gcv_params)
        print('the other params!')
    return model

def lgb_model_run(df, label):
    params_init = get_params_init()
    # 初始模型
    model_lgb = lgb.LGBMClassifier(**params_init)
    print(model_lgb.get_params())
    model_lgb = grid_params_model(model_lgb, df, label)
    return model_lgb