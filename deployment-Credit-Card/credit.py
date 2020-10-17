import streamlit as st

P = '''
<html>
<body>
<header class="masthead">
            <div class="w3-container d-flex h-100 align-items-center">
                <div class="mx-auto text-center">
                    <h1 class="text-black-50 text-uppercase"> Credit Account Prediction </h1>
                    <h1 class="text-black-60 mx-auto mt-2 mb-5 "><i> Credit Card Default Prediction </i></h1>
                </div>
            </div>
        </header>
    </body>
</html>'''

st.markdown(P, unsafe_allow_html=True)

st.write(' ### **Enter Casual Details** ')

L = st.text_input("Limit Balance", 0)
CHOICES = {1: "Graduate School", 2: "University", 3: "High School", 4 : "Other"}


def format_func(E):
    return CHOICES[E]


E = st.selectbox("Education", options=list(CHOICES.keys()), format_func=format_func)


CHOICE = {1: "Married", 2: "Single", 3: "Other"}

def ffunc(M):
    return CHOICE[M]

M = st.selectbox("Marital Status", options=list(CHOICE.keys()), format_func=ffunc)

age = st.text_input("Age (in years)", 0)


choice = {-2: "Account Started that month with zero balance and never used", -1: "Account had a balance that was paid in full", 0: "Atleast the minimum payments was made, but the entire balance wasn't paid", 1 : "Payment Delay for 1 month", 2 : "Payment Delay for 2 months", 3 : "Payment Delay for 3 months", 4 : "Payment Delay for 4 months", 5 : "Payment Delay for 5 months", 6 : "Payment Delay for 6 months", 7 : "Payment Delay for 7 months", 8 :"Payment Delay for 8 months"}

def formatfunc(pay1):
    return choice[pay1]

pay1 = st.selectbox("Payment Status of last month", options=list(choice.keys()), format_func=formatfunc)

st.write(' ### **Enter Past 6 Months Bill Amout Details** ')

lm1 = st.text_input("Last Month Bill Amount", 0)
lm2 = st.text_input("2nd Last Month Bill Amount", 0)
lm3 = st.text_input("3rd Last Month Bill Amount", 0)
lm4 = st.text_input("4th Last Month Bill Amount", 0)
lm5 = st.text_input("5th Last Month Bill Amount", 0)
lm6 = st.text_input("6th Last Month Bill Amount", 0)
st.write(' ### **Enter Past 6 Months Amout Paid Details** ')

am1 = st.text_input("Amount Paid in Last Month", 0)
am2 = st.text_input("Amount Paid in 2nd Last Month", 0)
am3 = st.text_input("Amount Paid in 3rd Last Month", 0)
am4 = st.text_input("Amount Paid in 4th Last Month", 0)
am5 = st.text_input("Amount Paid in 5th Last Month", 0)
am6 = st.text_input("Amount Paid in 6th Last Month", 0)


button = st.button('Predict')



l = [[int(L), E, M, int(age), pay1, int(lm1), int(lm2), int(lm3), int(lm4), int(lm5), int(lm6), int(am1), int(am2), int(am3), int(am4), int(am5), int(am6) ]]

 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('cleaned_data.csv')
features_response = df.columns.tolist()
items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'EDUCATION_CAT', 'graduate school', 'high school', 'none',
            'others', 'university']
features_response = [item for item in features_response if item not in items_to_remove]
rf = RandomForestClassifier\
        (n_estimators=200, criterion='gini', max_depth=9,
        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
        max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
        min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1,
        random_state=4, verbose=1, warm_start=False, class_weight=None)
rf.fit(df[features_response[:-1]].values, df['default payment next month'].values)

if button:
    pred = rf.predict(l)[0]
    pos_prob = rf.predict_proba(l)[0][1] * 100
    neg_prob = rf.predict_proba(l)[0][0] * 100
    if pred == 1:
        st.error(' **The Account will be defaulted with the probability of {:.4}%. ** '.format(pos_prob))
    else:
        st.success(' **The Account will not be defaulted with the probability of {:.4}%. ** '.format(neg_prob))
        

  
        
        

    
    
    