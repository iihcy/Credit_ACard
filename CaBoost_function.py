# coding:utf-8
# @Author:iihcy
import catboost as cat
from sklearn.grid_search import GridSearchCV

# 网格搜索策略--CatBoost
def get_params_init():
    params_init = {
        'iterations': 10,
        'cat_features': [0, 1, 2],
        'loss_function': 'Logloss',
        'depth': 5,
        'subsample': 0.7,
        'learning_rate': 0.1,
        'l2_leaf_reg': 0.1,
    }
    return params_init

def get_tree_params():
    params = {
        'iterations': [10, 50, 100, 150, 200],
        'depth': [5, 6, 7, 8],
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

def get_l2_params():
    params = {
        'l2_leaf_reg': [0.1, 0.5, 1, 2],
    }
    return params

def grid_params_model(model, df, label):
    # 参数组合
    grid_params = [get_tree_params(), get_subsample_params(), get_learn_params(), get_l2_params()]
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

def cat_model_run(df, label):
    params_init = get_params_init()
    # 初始模型
    model_cb = cat.CatBoostClassifier(**params_init)
    print(model_cb.get_params())
    model_cb = grid_params_model(model_cb, df, label)
    return model_cb