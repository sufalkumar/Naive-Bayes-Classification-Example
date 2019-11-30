#!/usr/bin/env python
# coding: utf-8

# In[20]:


# Assigning features and label variables
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
humidity=['H','H','H','H','N','N','N','H','N','N','N','H','N','H']
windy=['F','T','F','F','F','T','T','F','F','F','T','T','F','T']
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


# In[21]:


# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
wheather_encoded=le.fit_transform(weather)
print (wheather_encoded)


# In[23]:


# Converting string labels into numbers
temp_encoded=le.fit_transform(temp)
humidity_encoded=le.fit_transform(humidity)
windy_encoded=le.fit_transform(windy)
label=le.fit_transform(play)
print ("Temp:",temp_encoded)
print ("Humidity:",humidity_encoded)
print ("Windy:",windy_encoded)
print ("Play:",label)


# In[24]:


#Combinig weather and temp into single listof tuples
#features=zip(wheather_encoded,temp_encoded)
c = lambda wheather_encoded, temp_encoded, humidity_encoded, windy_encoded: [list(c) for c in zip(wheather_encoded,temp_encoded,humidity_encoded,windy_encoded)]
features = c(wheather_encoded, temp_encoded, humidity_encoded, windy_encoded)

print (list(features))
print(type(features))
print(type(label))


# In[26]:


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(features,label)

#Predict Output
predicted= model.predict([[2,1,0,0]]) # 0:Overcast, 2:Mild
print ("Predicted Value:", predicted)


# In[ ]:





# In[ ]:




