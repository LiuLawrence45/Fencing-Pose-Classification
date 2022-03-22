import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('coords.csv')
x = df.drop('class', axis = 1) #features
y = df['class'] #target value

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1234)


#----------- actual training now
from sklearn.pipeline import make_pipeline   #making machine learning pipeline - this will be used for training and scaling 
from sklearn.preprocessing import StandardScaler #normalizes the data, onto a level basis, so all features are equal


from sklearn.linear_model import LogisticRegression, RidgeClassifier      #different classification algorithms ... 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier      #different classification algorithms ... 

pipelines = {
    'lr': make_pipeline(StandardScaler(), LogisticRegression()),
    'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(x_train, y_train)
    fit_models[algo] = model



#------------ testing and serializing model
from sklearn.metrics import accuracy_score
import pickle

for algo, model in fit_models.items():
    yhat = model.predict(x_test)
    print(algo, accuracy_score(y_test, yhat))

# fit_models['rf'].predict(x_test)
with open('body_language.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)