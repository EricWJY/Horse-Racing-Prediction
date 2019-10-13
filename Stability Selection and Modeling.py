from sklearn import preprocessing
#Logistic Modeling
#Data transforming
for col in all_cate_var:
    dummy = pd.get_dummies(binary_data[col],prefix = col)
    dummy.drop(dummy.columns[0],axis = 1,inplace = True)
    binary_data.drop(col,axis = 1,inplace = True)
    binary_data = pd.concat([binary_data, dummy], axis = 1)

train_x = np.array(binary_data.drop(columns=['outcome']))
train_y = np.array(binary_data['outcome']).astype('int')

#Feature Selection
from stability_selection import StabilitySelection
from sklearn.svm import LinearSVC

lsvc = LinearSVC()
selector = StabilitySelection(base_estimator = lsvc).fit(train_x, train_y)
train_x_name = binary_data.drop(columns=['outcome']).columns
score_table = pd.DataFrame(np.mean(selector.stability_scores_,axis = 1),train_x_name,columns = ['Scores'])
score_table = score_table.sort_values(['Scores'],ascending=False)[0:30]

train_x_varselect = binary_data[train_x_name[selector.get_support()]]

#Modeling
from sklearn.linear_model import LogisticRegression
Logistic_model = LogisticRegression()
Logistic_model.fit(train_x_varselect, train_y)

#CV
from sklearn.model_selection import cross_val_score
Logistic_scores = np.mean(cross_val_score(Logistic_model, train_x_varselect, train_y))

###############################################################################################

#Tree Model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

#Data transforming
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for col in all_cate_var:
    binary_data[col] = encoder.fit_transform(binary_data[col])

train_x = np.array(binary_data.drop(columns=['outcome']))
train_y = np.array(binary_data['outcome']).astype('int')

#DecisionTree
DecisionTree_model = DecisionTreeClassifier(max_depth=4)
DecisionTree_model.fit(train_x, train_y)
DecisionTree_scores = np.mean(cross_val_score(DecisionTree_model, train_x, train_y))

#RandomForest
RandomForest_model = RandomForestClassifier(n_estimators=50, max_depth=5)
RandomForest_model.fit(train_x, train_y)
RandomForest_scores = np.mean(cross_val_score(RandomForest_model, train_x, train_y))

Parameters = []
for max_depth in range(1,11):
    for n_estimators in range(1,31):
        RandomForest_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        RandomForest_model.fit(train_x, train_y)
        RandomForest_scores = np.mean(cross_val_score(RandomForest_model, train_x, train_y))
        Parameters.append(max_depth)
        Parameters.append(n_estimators)
        Parameters.append(RandomForest_scores)
score_table = pd.DataFrame(np.array(Parameters).reshape([-1,3]),columns=['max_depth','n_estimators','scores'])
score_table = score_table.sort_values(['scores'],ascending=False)[0:10]

#ExtraTree
ExtraTrees_model = ExtraTreesClassifier()
ExtraTrees_model.fit(train_x, train_y)
ExtraTrees_scores = np.mean(cross_val_score(ExtraTrees_model, train_x, train_y))

Parameters = []
for max_depth in range(1,11):
    for n_estimators in range(1,31):
        ExtraTrees_model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth)
        ExtraTrees_model.fit(train_x, train_y)
        ExtraTrees_scores = np.mean(cross_val_score(ExtraTrees_model, train_x, train_y))
        Parameters.append(max_depth)
        Parameters.append(n_estimators)
        Parameters.append(ExtraTrees_scores)
score_table = pd.DataFrame(np.array(Parameters).reshape([-1,3]),columns=['max_depth','n_estimators','scores'])
score_table = score_table.sort_values(['scores'],ascending=False)[0:10]

#AdaBoosting
DecisionTree_model = DecisionTreeClassifier(max_depth=3)
AdaBoost_model = AdaBoostClassifier(base_estimator=DecisionTree_model)
AdaBoost_model.fit(train_x, train_y)
AdaBoost_scores = np.mean(cross_val_score(AdaBoost_model, train_x, train_y))

Parameters = []
for max_depth in range(1,11):
    for n_estimators in range(1,31):
        DecisionTree_model = DecisionTreeClassifier(max_depth=max_depth)
        AdaBoost_model = AdaBoostClassifier(base_estimator=DecisionTree_model,n_estimators=n_estimators)
        AdaBoost_model.fit(train_x, train_y)
        AdaBoost_scores = np.mean(cross_val_score(AdaBoost_model, train_x, train_y))
        Parameters.append(max_depth)
        Parameters.append(n_estimators)
        Parameters.append(AdaBoost_scores)
score_table = pd.DataFrame(np.array(Parameters).reshape([-1,3]),columns=['max_depth','n_estimators','scores'])
score_table = score_table.sort_values(['scores'],ascending=False)[0:10]

#GBDT
GBDT_model = GradientBoostingClassifier()
GBDT_model.fit(train_x, train_y)
GBDT_scores = np.mean(cross_val_score(GBDT_model, train_x, train_y))

Parameters = []
for max_depth in range(1,11):
    for n_estimators in range(1,31):
        GBDT_model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
        GBDT_model.fit(train_x, train_y)
        GBDT_scores = np.mean(cross_val_score(GBDT_model, train_x, train_y))
        Parameters.append(max_depth)
        Parameters.append(n_estimators)
        Parameters.append(GBDT_scores)
score_table = pd.DataFrame(np.array(Parameters).reshape([-1,3]),columns=['max_depth','n_estimators','scores'])
score_table = score_table.sort_values(['scores'],ascending=False)[0:10]

###############################################################################################

#SVM
#Data transforming
#[1] One-Hot
for col in all_cate_var:
    dummy = pd.get_dummies(binary_data[col],prefix = col)
    dummy.drop(dummy.columns[0],axis = 1,inplace = True)
    binary_data.drop(col,axis = 1,inplace = True)
    binary_data = pd.concat([binary_data, dummy], axis = 1)

#[2] StandardScaler
from sklearn.preprocessing import StandardScaler
SC = StandardScaler()
binary_data[all_con_var] = SC.fit_transform(binary_data[all_con_var])

train_x = np.array(binary_data.drop(columns=['outcome']))
train_y = np.array(binary_data['outcome']).astype('int')

#Feature Selection
from stability_selection import StabilitySelection
selector = StabilitySelection().fit(train_x, train_y)
train_x_name = binary_data.drop(columns=['outcome']).columns
score_table = pd.DataFrame(np.mean(selector.stability_scores_,axis = 1),train_x_name,columns = ['Scores'])
score_table = score_table.sort_values(['Scores'],ascending=False)[0:30]

train_x_varselect = binary_data[train_x_name[selector.get_support()]]

from sklearn.svm import SVC
#SVM模型拟合
SVM_model = SVC()
SVM_model.fit(train_x_varselect, train_y)
SVM_scores = np.mean(cross_val_score(SVM_model, train_x_varselect, train_y))

###############################################################################################

#Neural Network
#Data Transforming
#[1] One-Hot
for col in [all_cate_var+['outcome']]:
    dummy = pd.get_dummies(binary_data[col],prefix = col)
    binary_data.drop(col,axis = 1,inplace = True)
    binary_data = pd.concat([binary_data, dummy], axis = 1)

#[2] MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
MMS = MinMaxScaler()
binary_data[all_con_var] = MMS.fit_transform(binary_data[all_con_var])

train_x = np.array(binary_data.drop(columns=['outcome_0','outcome_1']))
train_y = np.array(binary_data['outcome_1']).astype('int')

#Feature Selection
from stability_selection import StabilitySelection
from sklearn.svm import LinearSVC

lsvc = LinearSVC()
selector = StabilitySelection(base_estimator = lsvc).fit(train_x, train_y)

#Create importance score table
train_x_name = binary_data.drop(columns=['outcome_0','outcome_1']).columns
score_table = pd.DataFrame(np.mean(selector.stability_scores_,axis = 1),train_x_name,columns = ['Scores'])
score_table = score_table.sort_values(['Scores'],ascending=False)[0:30]

#Select useful attributes
train_x_varselect = np.array(binary_data[train_x_name[selector.get_support()]])
train_y = np.array(binary_data[['outcome_0','outcome_1']]).astype('int')

#Modeling
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential

train_X,test_X, train_Y, test_Y = train_test_split(train_x_varselect,train_y)

model = Sequential()
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(2))
model.add(Activation('softmax'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

accuracy = []
for i in range(100):
    model.fit(train_X, train_Y, epochs=1, batch_size=1000)
    eval = model.evaluate(test_X, test_Y, verbose=0)
    accuracy.append([model.evaluate(train_X, train_Y, verbose=0)[1],eval[1]])
accuracy = np.array(accuracy)

#Plot the training process
import matplotlib.pyplot as plt

X = np.linspace(1,100,100,endpoint=True)
plt.plot(X,accuracy[:,0],label="training set")
plt.plot(X,accuracy[:,1],label="testing set")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()