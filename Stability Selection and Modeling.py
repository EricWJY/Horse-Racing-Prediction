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