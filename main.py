import pandas as pd
import numpy as np
import lightgbm as lgb ##Native lgbm
import optuna
from optuna.samplers import TPESampler ##Optuna Hp tuning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def load_data(data_path:str) -> pd.DataFrame:
    """ Loads a csv file and returns a pandas dataframe"""
    return pd.read_csv(data_path)


def encode_labels_for_training(df:pd.DataFrame):

    """Explicitly encode the labels. reproducible and better than sklearn's label encoder """

    label_encoder = dict()
    for i in range(len(df['category'].unique())):
        label_encoder[df['category'].unique()[i]] = i

    return label_encoder

def preprocess_data(df:pd.DataFrame):

    """Uses TF-IDF vectorizer for encodings"""

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', lowercase=True, max_features=5000)  ##USe this for inference
    X = vectorizer.fit_transform(df['transaction'])   ##Trainable data
    y = np.array(df['encoded_label']) 
    return X, y, vectorizer

##HP tuning logic using Optuna
def define_objective(train_x, valid_x, train_y, valid_y, num_classes):
    """
    HP tuning

    Args:-
        train_x, train_y  - Training datasets
        valid_x, valid_y - Validation datasets
        num_classes - Num of categories in the training data
    
    """
    def objective(trial):

        #search space
        params = {
        'objective': 'multiclass',
        'num_class': len(set(train_y)),
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 1e-2, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.2, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 1), 
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 1)
        }


        ##lgbm datasets
        train_data = lgb.Dataset(train_x, label=train_y)
        valid_data = lgb.Dataset(valid_x, label=valid_y)

        ##model training with early stopping
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
        )

        trial.set_user_attr("best_iteration", model.best_iteration)

        preds = model.predict(valid_x, num_iteration=model.best_iteration).argmax(axis=1) #pick the most relevant category - greedy sampling
        return f1_score(valid_y, preds, average='weighted') ##Due to class imbalance optimize a optuna trial for weighted f1 score

    return objective


def train_final_model(X,y, best_params, best_iteration):
    """
    Train the final model with best hps
    Args:- 
        X,y - Training + validation datasets - This is to not waste the validation dataset
        best_params - best hps from tuning
        best_iteration - best num_boost_round from early stopping - to avoid overfitting

    Returns:-
        final_model - a lgbm booster object

    """

    best_params.update({
        'objective': 'multiclass',
        'num_class': len(set(y)),
        'metric': 'multi_logloss'
    })

    return lgb.train(best_params, lgb.Dataset(X, label=y), num_boost_round=best_iteration)


def predict_on_test(final_model, test_df, label_encoder, vectorizer, description_col='transaction'):

    """
        Predict on test set
        Args:- 
            final_model - Model obtained after hp tuning
            test_df - Test dataframe
            label_encoder - A key value dictionary of labels
            vectorizer - TF-IDF vectorizer
            description_col - I/p column that contains transaction's descirption
    
    """

    tokens = vectorizer.transform(test_df[description_col])
    pred = final_model.predict(tokens).argmax(axis=1)
    reverse_label_encoder = { v:k for k,v in label_encoder.items()}
    clean_output = [reverse_label_encoder[ele] for ele in pred]
    test_df['category'] = clean_output

    return test_df



##Training function
def train():

    #Load the training data and encode labels for the same
    train_df = load_data("./data/ds_project_train_v1.csv")
    label_encoder = encode_labels_for_training(train_df)
    train_df['encoded_label'] = train_df['category'].apply(lambda x:label_encoder.get(x))

    ##Encode the transaction descripton
    X,y,vectorizer = preprocess_data(train_df)
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    #Hp tuning
    objective = define_objective(train_x, valid_x, train_y, valid_y, len(set(y)))
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(objective, n_trials=5)

    print("Best params:", study.best_params)

    ##Final model fitting
    final_model = train_final_model(X,y,study.best_params, study.best_trial.user_attrs["best_iteration"])
    test_df = load_data("./data/ds_project_test_v1.csv")

    #Predictions on test set
    test_predictions = predict_on_test(final_model, test_df, label_encoder, vectorizer)
    test_predictions.to_csv('./predictions.csv', index=False)





if __name__ == '__main__':
     train()