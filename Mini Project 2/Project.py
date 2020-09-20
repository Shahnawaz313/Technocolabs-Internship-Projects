import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import graphviz
from sklearn.model_selection import KFold
from sklearn import metrics



st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Technocolabs Internship Mini Project 2 by Mohammed Shahnawaz")
@st.cache(persist = True)
def load_data():
    df = pd.read_excel('Data set/default_of_credit_card_clients.xls',)
    return df
df = load_data()
if st.checkbox("Task 1 of Internship: Data Exploration and data cleaning"):
    st.write(df)
    np.random.seed(seed=24)
    random_integers = np.random.randint(low=1,high=5,size=100)
    is_equal_to_3 = random_integers == 3
    sum(is_equal_to_3)
    id_counts = df['ID'].value_counts()
    dupe_mask = id_counts == 2
    dupe_ids = id_counts.index[dupe_mask]
    dupe_ids = list(dupe_ids)
    duplen = len(dupe_ids)
    Orilen = len(df)
    st.write("Duplicate Entries:",duplen,"Original Entries:",Orilen)
    df.loc[df['ID'].isin(dupe_ids[0:3]),:].head(10)
    df_zero_mask = df == 0
    feature_zero_mask = df_zero_mask.iloc[:,1:].all(axis=1)
    sum(feature_zero_mask)
    df_clean_1 = df.loc[~feature_zero_mask,:].copy()
    df_clean_1['ID'].nunique()
    valid_pay_1_mask = df_clean_1['PAY_1'] != 'Not available'
    df_clean_2 = df_clean_1.loc[valid_pay_1_mask,:].copy()
    df_clean_2['PAY_1'] = df_clean_2['PAY_1'].astype('int64')
    df_clean_2[['PAY_1', 'PAY_2']].info()
    df_clean_2['EDUCATION'].replace(to_replace=[0, 5, 6], value=4, inplace=True)
    df_clean_2['EDUCATION'].value_counts()
    df_clean_2['MARRIAGE'].replace(to_replace=0, value=3, inplace=True)
    df_clean_2['MARRIAGE'].value_counts()
    df_clean_2['EDUCATION_CAT'] = 'none'
    df_clean_2[['EDUCATION', 'EDUCATION_CAT']].head(10)
    cat_mapping = {
    1: "graduate school",
    2: "university",
    3: "high school",
    4: "others"
    }
    df_clean_2['EDUCATION_CAT'] = df_clean_2['EDUCATION'].map(cat_mapping)
    df_clean_2[['EDUCATION', 'EDUCATION_CAT']].head(10)
    edu_ohe = pd.get_dummies(df_clean_2['EDUCATION_CAT'])
    edu_ohe.head(10)
    df_with_ohe = pd.concat([df_clean_2, edu_ohe], axis=1)
    df_with_ohe[['EDUCATION_CAT', 'graduate school','high school', 'university', 'others']].head(10)
    df_with_ohe.to_csv('cleaned_data.csv', index=False)
    cleaned_data = pd.read_csv("cleaned_data.csv")
    st.subheader("cleaned_data after removing Duplicate Entries")
    #st.subheader("cleaned_data after removing Duplicate Entries")
    #st.subheader("cleaned_data")
    st.write(cleaned_data)
    st.write(len(cleaned_data))
if st.checkbox("Task 2 of Internship: Exploration of remaining financial Insights"):
    bill_feats=['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    pay_amt_feats=['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    #st.subheader("Describing the data")
    st.subheader("Describing the data")
    desc = cleaned_data[bill_feats].describe()
    st.write(desc)
    st.markdown("**visualizing the bill amount features using a 2 by 3 grid of histogram plots.**")
    cleaned_data[bill_feats].hist(bins=20,layout=(2,3))
    st.pyplot()
    st.markdown("**histogram of the bill payment features similar to the bill amount features,**")
    cleaned_data[pay_amt_feats].hist(layout=(2,3), xrot=30)
    st.pyplot()
    # Create Boolean mask
    pay_zero_mask=cleaned_data[pay_amt_feats]==0
    st.markdown("**histograms of logarithmic transformations of the non-zero payments.**")
    cleaned_data[pay_amt_feats][~pay_zero_mask].apply(np.log10).hist(layout=(2,3))
    st.pyplot()

if st.checkbox("Task 3 Performing LogisticRegression"):


    X_train , X_test, y_train, y_test = train_test_split(cleaned_data['LIMIT_BAL'].values.reshape(-1,1), cleaned_data['default payment next month'].values, test_size = 0.2, random_state = 24)
    lg = LogisticRegression()
    lg.C = 0.1
    trained = lg.fit(X_train ,y_train)
    pred = trained.predict_proba(X_test)
    pos_prob =pred[:,1]
    fpr, tpr, threshold = roc_curve(y_test,pos_prob)
    roc_auc_score(y_test, pos_prob)
    plt.plot(fpr, tpr, '*-')
    plt.plot([0,1],[0,1],'--')
    plt.legend(['Logistic Regression', 'Random Chance'])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.show()
    st.pyplot()
    precision, recall, _ = precision_recall_curve(y_test, pos_prob)
    st.markdown("Plot of the precision-recall curve")
    plt.plot(recall, precision, 'r^-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('precision_recall curve')
    plt.show()
    st.pyplot()
    st.markdown("calculated the area under the precision-recall curve.")
    auc_score = auc(recall, precision)
    auc_score
    train_pred = trained.predict_proba(X_train)
    rs = roc_auc_score(y_train, train_pred[:,1])
    st.markdown("Finally, recalculate the ROC AUC, except this time do it for the training data.")
    rs
    st.markdown("How is this different, conceptually and quantitatively, from your earlier calculation? The 'roc_auc_score' for the training data is 0.6182918113358344 and the 'roc_auc_score' for testing data is 0.6201990844642832 .")

if st.checkbox("Task 4 of Internship"):
    st.header("Task 4 of Internship Fitting a logistic regression model")
    # Define the sigmoid function
    def sigmoid(X):
        f = 1 / (1 + np.exp(-X))
        return f

    X = cleaned_data.loc[:,['PAY_1', 'LIMIT_BAL']]
    y = cleaned_data['default payment next month']
    X_train , X_test, y_train, y_test = train_test_split(X,y, random_state = 24, test_size = 0.2)
    lr = LogisticRegression()
    # Fit the logistic regression model on training data
    lr_fit = lr.fit(X_train,y_train)
    # Make predictions using `.predict()`
    lr_fit.predict(X_test)
    # Find class probabilities using `.predict_proba()`
    lr_fit.predict_proba(X_test)
    # Add column of 1s to features
    X['intercept_col'] = 1
    st.markdown("added a column of 1s to features, to multiply by the intercept.")
    st.write(X)
    # Get coefficients and intercepts from trained model
    coef1 = lr_fit.coef_[0][0]
    coef2 = lr_fit.coef_[0][1]
    intercept = lr_fit.intercept_
    # Manually calculate predicted probabilities
    fun = intercept * X['intercept_col'] + coef1 * X['PAY_1'] + coef2 * X['LIMIT_BAL']
    manually_predicted_probabilities = sigmoid(fun)
    # Manually calculate predicted classes
    trainx, testx, trainy, testy = train_test_split(X,y, random_state = 24, test_size = 0.5)
    fit = lr.fit(trainx, trainy)
    # Compare to scikit-learn's predicted classes
    pred_0 = fit.predict(testx)
    skrs = roc_auc_score(testy, pred_0 )
    st.write("Using scikit-learn's predicted probabilities to calculate ROC AUC",skrs)
    # Use manually calculated predicted probabilities to calculate ROC AUC
    mrs = roc_auc_score(testy, pred_0)
    st.write("manually calculated predicted probabilities to calculate ROC AUC",mrs)

if st.checkbox("Task 5 of Internship"):
    st.header("Task 5 of Internship: Cross Validation and Feature Engineering ")
    # Create features list
    features_list = ['LIMIT_BAL','EDUCATION','MARRIAGE','AGE','PAY_1','BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    X = cleaned_data.loc[:,features_list]
    y = cleaned_data['default payment next month']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 24)
    scaler = MinMaxScaler().fit(X,y)
    scalar1 = MinMaxScaler().fit(X)
    x = scalar1.transform(X)
    lr = LogisticRegression(solver = 'saga', penalty = 'l1', max_iter = 1000)
    pipeline = Pipeline(steps=[('scaler', scaler), ('model', lr)])
    #pipeline
    # Use `get_params`
    #pipeline.get_params
    # View what `model__C` is set to currently
    #pipeline.get_params()
    # Change `model__C` to 2
    pipeline.set_params(model__C = 2)
    #Then, create a smaller range of C values to test with cross-validation, as these models will take longer to train and test with more data than our previous activities.

    #Use C_vals = [$10^2$, $10$, $1$, $10^{-1}$, $10^{-2}$, $10^{-3}$].
    C_val_exponents = np.linspace(2,-3,6)
    C_vals = np.float(10)**C_val_exponents
    #C_vals
    k_folds = StratifiedKFold(n_splits = 4, random_state = 1)

    def cross_val_C_search_pipe(k_folds, C_vals, model, X, y):
        n_folds = k_folds.n_splits
        cv_train_roc_auc = np.empty((n_folds, len(C_vals)))
        cv_test_roc_auc = np.empty((n_folds, len(C_vals)))
        cv_test_roc = [[]]*len(C_vals)
        for c_val_counter in range(len(C_vals)):
            #Set the C value for the model object
            model.C = C_vals[c_val_counter]
            #Count folds for each value of C
            fold_counter = 0
            #Get training and testing indices for each fold
            for train_index, test_index in k_folds.split(X, y):
                #Subset the features and response, for training and testing data for
                #this fold
                X_cv_train, X_cv_test = X[train_index], X[test_index]
                y_cv_train, y_cv_test = y[train_index], y[test_index]
                #Fit the model on the training data
                model.fit(X_cv_train, y_cv_train)
                #Get the training ROC AUC
                y_cv_train_predict_proba = model.predict_proba(X_cv_train)
                cv_train_roc_auc[fold_counter, c_val_counter] = roc_auc_score(y_cv_train, y_cv_train_predict_proba[:,1])
                #Get the testing ROC AUC
                y_cv_test_predict_proba = model.predict_proba(X_cv_test)
                cv_test_roc_auc[fold_counter, c_val_counter] = roc_auc_score(y_cv_test, y_cv_test_predict_proba[:,1])
                #Testing ROC curves for each fold
                this_fold_roc = roc_curve(y_cv_test, y_cv_test_predict_proba[:,1])
                cv_test_roc[c_val_counter].append(this_fold_roc)
                #Increment the fold counter
                fold_counter += 1
                #Indicate progress
                print('Done with C = {}'.format(model.C))
        return cv_train_roc_auc, cv_test_roc_auc, cv_test_roc



    cv_train_roc_auc, cv_test_roc_auc, cv_test_roc = cross_val_C_search_pipe(k_folds, C_vals, pipeline, x, y)


    st.markdown("Plotted the average training and testing ROC AUC across folds, for each np.log(C_vals) value.")
    for this_fold in range(4):
        plt.plot(C_val_exponents, cv_train_roc_auc[this_fold], '-o', label='Training fold {}'.format(this_fold+1))
        plt.plot(C_val_exponents, cv_test_roc_auc[this_fold], '-x', label='Testing fold {}'.format(this_fold+1))
    plt.ylabel('ROC AUC')
    plt.xlabel('log$_{10}$(C)')
    plt.legend(loc = [1.1, 0.2])
    st.pyplot()
    plt.title('Cross validation scores for each fold')
    plt.plot(C_val_exponents, np.mean(cv_train_roc_auc, axis=0), '-o',label='Average training score')
    plt.plot(C_val_exponents, np.mean(cv_test_roc_auc, axis=0), '-x',label='Average testing score')
    plt.ylabel('ROC AUC')
    plt.xlabel('log$_{10}$(C)')
    plt.legend()
    plt.title('Cross validation scores averaged over all folds')
    st.pyplot()

    poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    # Using the new features, make a 80:20 train/test split using a random seed of 24.**
    t = poly_features.fit(X,y)
    xx = poly_features.fit(X).transform(X)
    Pipeline2 = Pipeline(steps=[('t', t), ('model', lr)])
    #Pipeline
    # Call the cross_val_C_search_pipe() function using the new training data.
    # All other parameters should remain the same.
    # Note that this training may take a few minutes due to the larger number of features.
    cv_train_roc_auc, cv_test_roc_auc, cv_test_roc = cross_val_C_search_pipe(k_folds, C_vals, Pipeline2, xx, y)

    # Plot the average training and testing ROC AUC across folds, for each C value.
    plt.plot(C_val_exponents, np.mean(cv_train_roc_auc, axis=0), '-o',label='Average training score')
    st.pyplot()
    plt.plot(C_val_exponents, np.mean(cv_test_roc_auc, axis=0), '-x',label='Average testing score')
    st.pyplot()
    plt.ylabel('ROC AUC')
    plt.xlabel('log$_{10}$(C)')
    plt.legend()
    plt.title('Cross validation scores averaged over all folds')
    st.pyplot()
if st.checkbox("Task 6 of Internship"):
    st.header("Task 6 of Internship: Cross Validation Grid Search with Random Forest")

    features_response = cleaned_data.columns.tolist()
    items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university']
    features_response = [item for item in features_response if item not in items_to_remove]
                   #features_response
    X_train, X_test, y_train, y_test = train_test_split(
    cleaned_data[features_response[:-1]].values,
    cleaned_data['default payment next month'].values,
    test_size=0.2, random_state=24)
    rf = RandomForestClassifier(
    n_estimators=10, criterion='gini', max_depth=3,
    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
    max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1,
    random_state=4, verbose=0, warm_start=False, class_weight=None
    )
    params = {
        'max_depth': [3,6,9,12],
        'n_estimators': [10,50,100,200]
        }
    cv = GridSearchCV(rf, param_grid=params, scoring='roc_auc',n_jobs=None, iid=False, refit=True, cv=4, verbose=2,pre_dispatch=None, error_score=np.nan, return_train_score=True)
    cv.fit(X_train, y_train)
    cv_result = pd.DataFrame(cv.cv_results_)
    #cv_result
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    st.pyplot()
    axs[0].plot(cv_result['param_n_estimators'],
            cv_result['mean_fit_time'],
            '-o')
    axs[0].set_xlabel('Number of trees')
    axs[0].set_ylabel('Mean fit time (seconds)')
    axs[1].errorbar(cv_result['param_n_estimators'],
                cv_result['mean_test_score'],
                yerr=cv_result['std_test_score'])
    axs[1].set_xlabel('Number of trees')
    axs[1].set_ylabel('Mean testing ROC AUC $\pm$ 1 SD ')
    plt.tight_layout()
    st.pyplot()

    ax = plt.axes()
    ax.errorbar(cv_result['param_max_depth'],
            cv_result['mean_train_score'],
            yerr=cv_result['std_train_score'],
            label='Mean $\pm$ 1 SD training scores')
    ax.errorbar(cv_result['param_max_depth'],
            cv_result['mean_test_score'],
            yerr=cv_result['std_test_score'],
            label='Mean $\pm$ 1 SD testing scores')
    ax.legend()
    plt.xlabel('max_depth')
    plt.ylabel('ROC AUC')
    st.pyplot()

    #cv.best_params_
    # Create a 5x5 grid
    mean_test_score = cv_result['mean_test_score'].values.reshape(4,4)
    #mean_test_score
    # Set color map to `plt.cm.jet`
    xx, yy = np.meshgrid(range(5), range(5))
    # Visualize pcolormesh
    ax = plt.axes()
    pcolor_ex = ax.pcolormesh(xx, yy, mean_test_score, cmap=plt.cm.jet)
    plt.colorbar(pcolor_ex, label='Color scale')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    st.pyplot()
    # Create a dataframe of the feature names and importance
    feat_imp_df = pd.DataFrame({'Features': features_response[:-1], 'Importance': cv.best_estimator_.feature_importances_})
    #feat_imp_df
    # Sort values by importance
    feat_imp_df.sort_values('Importance', ascending = False)

if st.checkbox("Task 7 of Internship"):
    st.header("Task 7 of Internship: Deriving Financial Insights ")
    #mpl.rcParams['figure.dpi'] = 400
    df_orig = pd.read_excel('Data set/default_of_credit_card_clients.xls')
    df_zero_mask = df_orig == 0
    feature_zero_mask = df_zero_mask.iloc[:,1:].all(axis=1)
    st.write(sum(feature_zero_mask))
    df_clean = df_orig.loc[~feature_zero_mask,:].copy()
    df_clean['EDUCATION'].replace(to_replace=[0, 5, 6], value=4, inplace=True)
    #Should only be (1 = married; 2 = single; 3 = others).
    df_clean['MARRIAGE'].replace(to_replace=0, value=3, inplace=True)
    missing_pay_1_mask = df_clean['PAY_1'] == 'Not available'
    df_missing_pay_1 = df_clean.loc[missing_pay_1_mask,:].copy()
    df = pd.read_csv('cleaned_data.csv')
    features_response = df.columns.tolist()
    items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university']
    features_response = [item for item in features_response if item not in items_to_remove]
    X_train, X_test, y_train, y_test = \
    train_test_split(df[features_response[:-1]].values, df['default payment next month'].values,
    test_size=0.2, random_state=24)
    np.random.seed(seed=1)
    fill_values = [0, np.random.choice(X_train[:,4], size=(3021,), replace=True)]
    fill_strategy = ['mode', 'random']
    fig, axs = plt.subplots(1,2, figsize=(8,3))
    bin_edges = np.arange(-2,9)
    axs[0].hist(X_train[:,4], bins=bin_edges, align='left')
    axs[0].set_xticks(bin_edges)
    axs[0].set_title('Non-missing values of PAY_1')
    axs[1].hist(fill_values[-1], bins=bin_edges, align='left')
    axs[1].set_xticks(bin_edges)
    axs[1].set_title('Random selection for imputation')
    plt.tight_layout()
    st.pyplot()
    k_folds = KFold(n_splits=4, shuffle=True, random_state=1)
    rf = RandomForestClassifier\
    (n_estimators=200, criterion='gini', max_depth=9,
    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
    max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None,
    random_state=4, verbose=1, warm_start=False, class_weight=None)
    for counter in range(len(fill_values)):
        #Copy the data frame with missing PAY_1 and assign imputed values
        df_fill_pay_1_filled = df_missing_pay_1.copy()
        df_fill_pay_1_filled['PAY_1'] = fill_values[counter]

        #Split imputed data in to training and testing, using the same
        #80/20 split we have used for the data with non-missing PAY_1
        X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test = \
        train_test_split(
            df_fill_pay_1_filled[features_response[:-1]].values,
            df_fill_pay_1_filled['default payment next month'].values,
            test_size=0.2, random_state=24)

        #Concatenate the imputed data with the array of non-missing data
        X_train_all = np.concatenate((X_train, X_fill_pay_1_train), axis=0)
        y_train_all = np.concatenate((y_train, y_fill_pay_1_train), axis=0)

        #Use the KFolds splitter and the random forest model to get
        #4-fold cross-validation scores for both imputation methods
        imputation_compare_cv = cross_validate(rf, X_train_all, y_train_all, scoring='roc_auc',
                                       cv=k_folds, n_jobs=-1, verbose=1,
                                       return_train_score=True, return_estimator=True,
                                       error_score='raise-deprecating')

        test_score = imputation_compare_cv['test_score']
        print(fill_strategy[counter] + ' imputation: ' +
          'mean testing score ' + str(np.mean(test_score)) +
          ', std ' + str(np.std(test_score)))
    pay_1_df = df.copy()
    features_for_imputation = pay_1_df.columns.tolist()
    items_to_remove_2 = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university', 'default payment next month', 'PAY_1']
    features_for_imputation = [item for item in features_for_imputation if item not in items_to_remove_2]
    X_impute_train, X_impute_test, y_impute_train, y_impute_test = \
    train_test_split(
        pay_1_df[features_for_imputation].values,
        pay_1_df['PAY_1'].values,test_size=0.2, random_state=24)
    rf_impute_params = {'max_depth':[3, 6, 9, 12],
             'n_estimators':[10, 50, 100, 200]}
    cv_rf_impute = GridSearchCV(rf, param_grid=rf_impute_params, scoring='accuracy',
                            n_jobs=-1, iid=False, refit=True,
                            cv=4, verbose=2, error_score=np.nan, return_train_score=True)
    cv_rf_impute.fit(X_impute_train, y_impute_train)
    impute_df = pd.DataFrame(cv_rf_impute.cv_results_)
    pay_1_value_counts = pay_1_df['PAY_1'].value_counts().sort_index()
    y_impute_predict = cv_rf_impute.predict(X_impute_test)
    metrics.accuracy_score(y_impute_test, y_impute_predict)
    fig, axs = plt.subplots(1,2, figsize=(8,3))
    axs[0].hist(y_impute_test, bins=bin_edges, align='left')
    axs[0].set_xticks(bin_edges)
    axs[0].set_title('Non-missing values of PAY_1')
    axs[1].hist(y_impute_predict, bins=bin_edges, align='left')
    axs[1].set_xticks(bin_edges)
    axs[1].set_title('Model-based imputation')
    plt.tight_layout()
    st.pyplot()
    X_impute_all = pay_1_df[features_for_imputation].values
    y_impute_all = pay_1_df['PAY_1'].values
    rf_impute = RandomForestClassifier(n_estimators=100, max_depth=12)
    rf_impute.fit(X_impute_all, y_impute_all)
    df_fill_pay_1_model = df_missing_pay_1.copy()
    df_fill_pay_1_model['PAY_1'] = rf_impute.predict(df_fill_pay_1_model[features_for_imputation].values)
    X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test = \
    train_test_split(
        df_fill_pay_1_model[features_response[:-1]].values,
        df_fill_pay_1_model['default payment next month'].values,
        test_size=0.2, random_state=24)
    X_train_all = np.concatenate((X_train, X_fill_pay_1_train), axis=0)
    y_train_all = np.concatenate((y_train, y_fill_pay_1_train), axis=0)
    imputation_compare_cv = cross_validate(rf, X_train_all, y_train_all, scoring='roc_auc',
                                       cv=k_folds, n_jobs=-1, verbose=1,
                                       return_train_score=True, return_estimator=True,
                                       error_score='raise-deprecating')
    df_fill_pay_1_model['PAY_1'] = np.zeros_like(df_fill_pay_1_model['PAY_1'].values)
    X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test = \
    train_test_split(
        df_fill_pay_1_model[features_response[:-1]].values,
        df_fill_pay_1_model['default payment next month'].values,test_size=0.2, random_state=24)
    X_train_all = np.concatenate((X_train, X_fill_pay_1_train), axis=0)
    X_test_all = np.concatenate((X_test, X_fill_pay_1_test), axis=0)
    y_train_all = np.concatenate((y_train, y_fill_pay_1_train), axis=0)
    y_test_all = np.concatenate((y_test, y_fill_pay_1_test), axis=0)
    imputation_compare_cv = cross_validate(rf, X_train_all, y_train_all, scoring='roc_auc',
                                       cv=k_folds, n_jobs=-1, verbose=1,
                                       return_train_score=True, return_estimator=True,
                                       error_score='raise-deprecating')
    rf.fit(X_train_all, y_train_all)
    y_test_all_predict_proba = rf.predict_proba(X_test_all)
    roc_auc_score(y_test_all, y_test_all_predict_proba[:,1])
    thresholds = np.linspace(0, 1, 101)
    savings_per_default = np.mean(X_test_all[:, 5])
    cost_per_counseling = 7500
    effectiveness = 0.70
    n_pos_pred = np.empty_like(thresholds)
    cost_of_all_counselings = np.empty_like(thresholds)
    n_true_pos = np.empty_like(thresholds)
    savings_of_all_counselings = np.empty_like(thresholds)
    counter = 0
    for threshold in thresholds:
        pos_pred = y_test_all_predict_proba[:,1]>threshold
        n_pos_pred[counter] = sum(pos_pred)
        cost_of_all_counselings[counter] = n_pos_pred[counter] * cost_per_counseling
        true_pos = pos_pred & y_test_all.astype(bool)
        n_true_pos[counter] = sum(true_pos)
        savings_of_all_counselings[counter] = n_true_pos[counter] * savings_per_default * effectiveness
        counter += 1
    net_savings = savings_of_all_counselings - cost_of_all_counselings
    plt.plot(thresholds, cost_of_all_counselings)
    st.pyplot()
    plt.plot(thresholds, savings_of_all_counselings)
    st.pyplot()
    mpl.rcParams['figure.dpi'] = 400
    plt.plot(thresholds, net_savings)
    plt.xlabel('Threshold')
    plt.ylabel('Net savings (NT$)')
    plt.xticks(np.linspace(0,1,11))
    plt.grid(True)
    st.pyplot()
    max_savings_ix = np.argmax(net_savings)
    thresholds[max_savings_ix]
    net_savings[max_savings_ix]
    cost_of_default=sum(y_test_all) * savings_per_default
    st.write("Net savings:",net_savings[max_savings_ix]/cost_of_default)
    #net_savings[max_savings_ix]/len(y_test_all)
    plt.plot(cost_of_all_counselings/len(y_test_all), net_savings/len(y_test_all))
    plt.xlabel('Cost of all counseling per account')
    plt.ylabel('Net savings per account (NT$)')
    st.pyplot()
    plt.plot(thresholds, n_pos_pred/len(y_test_all))
    plt.xlabel("Thresholds")
    plt.ylabel("flag rate")
    st.pyplot()
    plt.plot(n_true_pos/len(y_test_all),np.divide(n_true_pos,n_pos_pred))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    st.pyplot()
    plt.plot(thresholds, np.divide(n_true_pos,n_pos_pred), label="Precision")
    plt.plot(thresholds, n_true_pos/sum(y_test_all), label="Recall")
    plt.xlabel("Thresholds")
    plt.legend()
    st.pyplot()
