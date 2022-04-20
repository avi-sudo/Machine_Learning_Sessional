import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
#from sklearn.linear_model import LogisticRegression

np.random.seed(1605006)

def read_Telco_Customer_Churn_Dataset():
    df=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv",skipinitialspace=True) #7043 samples #5174 no class, 1869 yes class
    #print(df.info())
    scaler=MinMaxScaler()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors = 'coerce')
    #print(df.dtypes)
    mean=df['TotalCharges'].mean()
    df.fillna(round(mean,2),inplace=True) #3828 in excel.....idx=3826
    
    y=df.drop(["customerID"],axis=1)
    y2=y["gender"].replace({'Male':1,'Female':0})
    y=y.drop(["gender"],axis=1)
    y=pd.concat([y,y2],axis=1)

    features_Yes_No=["Partner","Dependents","PhoneService","PaperlessBilling"]
    new_array=[]
    for i in features_Yes_No:
        y2=y[i].replace({'Yes':1,'No':0})
        new_array.append(y2)
        y=y.drop([i],axis=1)
    for i in new_array:
        y=pd.concat([y,i],axis=1)

    y3=y["Churn"].replace({'Yes':1,'No':-1})
    y=y.drop(["Churn"],axis=1)
    features_category=["MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaymentMethod"]
    features_scaling=["tenure","MonthlyCharges","TotalCharges"]
    for i in features_scaling:
        y[["scaled_"+i]]=scaler.fit_transform(y[[i]])
    
    y=y.drop(features_scaling,axis=1)
    #y=pd.get_dummies(data=y, columns=features_category)
    for i in features_category:
        y4=pd.get_dummies(y[i],prefix=i)
        y4=y4.iloc[ : , :-1]
        y=pd.concat([y,y4],axis=1)

    y=y.drop(features_category,axis=1)   
    y=pd.concat([y,y3],axis=1) 
    return y

def entropy(column):
    if 1 not in column.value_counts():
        return 0
        
    if -1 not in column.value_counts():
        return 0
    
    p=column.value_counts()[1]
    n=column.value_counts()[-1]
    total=p+n
    p1=p/total
    n1=n/total
    if p1==0:
        entropy=0
    elif n1==0:
        entropy=0
    else:
        entropy=-(p1*math.log(p1,2))-(n1*math.log(n1,2))
    #print(entropy)
    #print(df["Churn"].value_counts()[1]) #5174 no class, 1869 yes class
    #print(p,n)
    return entropy

def calculate_information_gain(df,attribute,class_name):
   
    output_entropy=entropy(df[class_name])
    unique_features = set()
    for var in df[attribute]:
        unique_features.add(var)
    total=len(df[attribute]) #p+n
    count=[]
    remainder=0
    
    for i in unique_features:
        cnt=0
        for j in df[attribute]:
            if i==j:
                cnt+=1
        count.append(cnt)
    feature_wise_dataset=[]
    for feature in unique_features:
        feature_wise_dataset.append(df.where(df[attribute]==feature).dropna()[class_name])
    
    
    for i in range(len(count)):
        p=count[i]/total #pk+nk/p+n
        
        remainder+=p*entropy(feature_wise_dataset[i])
    Gain=output_entropy-remainder
    #print(Gain)
    return Gain
def find_sorted_list(df,value):
    
    gain_attribute={}
    #print(df.columns[-1])
    
    y=df.drop([df.columns[-1]],axis=1)
    #print(y.head(1))

    for i in y.columns:
        gain=calculate_information_gain(df,i,df.columns[-1])
        gain_attribute[i]=gain
    sorted_gain_attribute=sorted(gain_attribute.items(), key=lambda x:x[1],reverse=True)
    top_attributes=[sorted_gain_attribute[x][0] for x in range(value)]
    
   # print(sorted_gain_attribute)
    #print(top_attributes)
    return top_attributes

def read_Telco_Customer_Churn_Dataset_for_weak_learner(value):
    df=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv",skipinitialspace=True) #19 active features
    y=df.drop(["customerID"],axis=1)

    if value > (len(y.columns)-1):
        print("Given Number of Features is overflowing.Please give valid feature count")
        exit(0)
    scaler=MinMaxScaler()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors = 'coerce')
    #print(df.dtypes)
    mean=df['TotalCharges'].mean()
    df.fillna(round(mean,2),inplace=True) #3828 in excel.....idx=3826
    
    
    y2=y["gender"].replace({'Male':1,'Female':0})
    y=y.drop(["gender"],axis=1)
    y=pd.concat([y,y2],axis=1)

    features_Yes_No=["Partner","Dependents","PhoneService","PaperlessBilling"]
    new_array=[]
    for i in features_Yes_No:
        y2=y[i].replace({'Yes':1,'No':0})
        new_array.append(y2)
        y=y.drop([i],axis=1)
    for i in new_array:
        y=pd.concat([y,i],axis=1)

    y3=y["Churn"].replace({'Yes':1,'No':-1})
    y=y.drop(["Churn"],axis=1)
    features_category=["MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaymentMethod"]
    features_scaling=["tenure","MonthlyCharges","TotalCharges"]
    for i in features_scaling:
        y[["scaled_"+i]]=scaler.fit_transform(y[[i]])
        y["scaled_"+i]=pd.cut(y["scaled_"+i],bins=2,labels=np.arange(2), right=False)
        
    
    y=y.drop(features_scaling,axis=1)
    y=pd.concat([y,y3],axis=1)
    
    top_attributes=find_sorted_list(y,value)
    #fetch top 8 attributes from old df to new data frame
    df2=pd.DataFrame()
    for i in top_attributes:
        df2=pd.concat([df2,y[i]],axis=1)
    #df2 is new data frame
    
    for i in top_attributes:
        if i in features_category:
            y4=pd.get_dummies(df2[i],prefix=i)
            y4=y4.iloc[ : , :-1]
            df2=pd.concat([df2,y4],axis=1)
            df2=df2.drop([i],axis=1)
    df2=pd.concat([df2,y3],axis=1)
    #print(df2.head(1))
    return df2

def read_adult_training_dataset():
    
    column_names=["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","output_label"]
    df = pd.read_csv("adult.data", sep=',',names=column_names,skipinitialspace=True)
    #df=pd.read_csv("adult_train.csv",names=column_names,skipinitialspace=True) #32561 samples, <=50K  24720,>50K   7841
    df=df.replace('?',df.mode().loc[0])
    #print(df["output_label"].value_counts())

    scaler=MinMaxScaler()
    features_scaling=["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
    for i in features_scaling:
        df[["scaled_"+i]]=scaler.fit_transform(df[[i]])
    
    df=df.drop(features_scaling,axis=1)
    
    y2=df["sex"].replace({'Male':1,'Female':0})
    df=df.drop(["sex"],axis=1)
    df=pd.concat([df,y2],axis=1)
    y3=df["output_label"].replace({'>50K':1,'<=50K':-1})
    df=df.drop(["output_label"],axis=1)
    
    features_category=["workclass","education","marital-status","occupation","relationship","race","native-country"]
    for i in features_category:
        y4=pd.get_dummies(df[i],prefix=i)
        if i != "native-country":
            y4=y4.iloc[ : , :-1]
        df=pd.concat([df,y4],axis=1)
    
    df=df.drop(features_category,axis=1) 
    #y4=pd.get_dummies(df["native-country"],prefix="native-country")
    #df=pd.concat([df,y4],axis=1)
    df=df.drop(["native-country_Holand-Netherlands"],axis=1)
    df=pd.concat([df,y3],axis=1)
    return df
def read_adult_test_dataset(type=0):
    
    column_names=["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","output_label"]
    df = pd.read_csv("adult.test", sep=',',names=column_names,skipinitialspace=True)
    #df=pd.read_csv("adult_test.csv",names=column_names,skipinitialspace=True) #16281 samples; <=50K. 12435  >50K.  3846
    #print(df.dtypes)
    
    df=df.iloc[1: , : ]
    #print(df["output_label"].value_counts())
    #df=df.replace(r'\?',np.NaN,regex=True)
   
    #df.fillna(value=df.mode().loc[0],inplace=True)
    df=df.replace('?',df.mode().loc[0])
    #print(df.columns)
    df['age'] = pd.to_numeric(df['age'],errors = 'coerce')
    
    scaler=MinMaxScaler()
    features_scaling=["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
    for i in features_scaling:
        df[["scaled_"+i]]=scaler.fit_transform(df[[i]])
        if type==1:
            df["scaled_"+i]=pd.cut(df["scaled_"+i],bins=2,labels=np.arange(2), right=False)
    df=df.drop(features_scaling,axis=1)
    
    y2=df["sex"].replace({'Male':1,'Female':0})
    df=df.drop(["sex"],axis=1)
    df=pd.concat([df,y2],axis=1)
    y3=df["output_label"].replace({'>50K.':1,'<=50K.':-1})
    
    df=df.drop(["output_label"],axis=1)
    
    features_category=["workclass","education","marital-status","occupation","relationship","race","native-country"]
    for i in features_category:
        y4=pd.get_dummies(df[i],prefix=i)
        if i != "native-country":
            y4=y4.iloc[ : , :-1]
        df=pd.concat([df,y4],axis=1)
        
    df=df.drop(features_category,axis=1) 
    #  Holand-Netherlands data is missing in native_country
    #df["native-country_Holand-Netherlands"]=np.zeros((df.count()["scaled_age"],1))
   
    df=pd.concat([df,y3],axis=1)
    return df

def read_adult_training_dataset_for_weak_learner(value): #10th active feature is native country
    column_names=["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","output_label"]
    df = pd.read_csv("adult.data", sep=',',names=column_names,skipinitialspace=True)
    #df=pd.read_csv("adult_train.csv",names=column_names,skipinitialspace=True) #32561 samples ,14 active features
    if value > (len(df.columns)-1):
        print("Given Number of Features is overflowing.Please give valid feature count")
        exit(0)
    scaler=MinMaxScaler()
    df=df.replace('?',df.mode().loc[0])

    scaler=MinMaxScaler()
    features_scaling=["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
    for i in features_scaling:
        df[["scaled_"+i]]=scaler.fit_transform(df[[i]])
        df["scaled_"+i]=pd.cut(df["scaled_"+i],bins=2,labels=np.arange(2), right=False)

    df=df.drop(features_scaling,axis=1)
    
    y2=df["sex"].replace({'Male':1,'Female':0})
    df=df.drop(["sex"],axis=1)
    df=pd.concat([df,y2],axis=1)
    y3=df["output_label"].replace({'>50K':1,'<=50K':-1})
    df=df.drop(["output_label"],axis=1)
   
    df=pd.concat([df,y3],axis=1)
    
    top_attributes=find_sorted_list(df,value)
    #fetch top value attributes from old df to new data frame
    df2=pd.DataFrame()
    for i in top_attributes:
        df2=pd.concat([df2,df[i]],axis=1)
    #df2 is new data frame
    features_category=["workclass","education","marital-status","occupation","relationship","race","native-country"]
    for i in top_attributes:
        if i in features_category:
            y4=pd.get_dummies(df2[i],prefix=i)
            if i != "native-country":
                y4=y4.iloc[ : , :-1]
            df2=pd.concat([df2,y4],axis=1)
            df2=df2.drop([i],axis=1)
            if i == "native-country":
                df2=df2.drop(["native-country_Holand-Netherlands"],axis=1) #missing feature in test data
                
    df2=pd.concat([df2,y3],axis=1)
    
    return df2

def read_adult_test_dataset_for_weak_learner(value): #10th active feature is native country
    test_df=pd.DataFrame()
    new_df=read_adult_test_dataset(1) #type 1 for binning scaled values
    training_df=read_adult_training_dataset_for_weak_learner(value)
    for i in training_df.columns:
        test_df=pd.concat([test_df,new_df[i]],axis=1)
    return test_df

def read_creditCard_dataset(value,type=0,top_features=30):
    df=pd.read_csv("creditcard.csv",skipinitialspace=True) #2,84,807 samples, 492 yes cls
    if top_features > (len(df.columns)-1):
        print("Given Number of Features is overflowing.Please give valid feature count")
        exit(0)
    df1=pd.DataFrame()
    df2=pd.DataFrame()
    
    yes_index=df[df[df.columns[-1]]==1].index
    no_index=df[df[df.columns[-1]]==0].index
    for i in yes_index:
        df1=df1.append(df.loc[i])
    sampled_no_index=np.random.choice(no_index,value,replace=False)
    
    for i in sampled_no_index:
        df2=df2.append(df.loc[i])

    df=df1.append(df2,ignore_index=True)
    
    scaler=MinMaxScaler()
    df.fillna(value=df.mean(),inplace=True)
    features_scaling=["V"+str(x) for x in range(1,29)]
    features_scaling.append("Time")
    features_scaling.append("Amount")
    for i in features_scaling:
        df[["scaled_"+i]]=scaler.fit_transform(df[[i]])
        if type==1:
            df["scaled_"+i]=pd.cut(df["scaled_"+i],bins=4,labels=np.arange(4), right=False)
    df=df.drop(features_scaling,axis=1)
    y=df["Class"].replace({0:-1})
    df=df.drop(["Class"],axis=1)
    
    df=pd.concat([df,y],axis=1)
    return df

def read_creditCard_Dataset_for_weak_learner(value,top_features):
    df=read_creditCard_dataset(value,1,top_features)
    
    top_attributes=find_sorted_list(df,top_features)
    #fetch top value attributes from old df to new data frame
    df3=pd.DataFrame()
    for i in top_attributes:
        df3=pd.concat([df3,df[i]],axis=1)
        df3[i]=pd.to_numeric(df3[i])
    
    return df3

def find_predicted_y_hat(Xw):
    return np.tanh(Xw)

def Logistic_Regression(X,Y,learning_rate,min_error,max_iteration):
    sample_count,features_count=X.shape
    #Y=Y.reshape(sample_count,1)
    #weight=np.zeros((features_count+1,1))
    weight=np.zeros((features_count,1))
    #dummy=np.ones((sample_count,1))
    #X=np.concatenate((dummy,X),axis=1)
    for i in range(max_iteration):
        Xw=np.dot(X,weight)
        y_hat=find_predicted_y_hat(Xw)
        
        #L2_loss=(np.sum((Y-y_hat) ** 2))/sample_count
        prediction_output=(y_hat>=0)*2 - 1
        Accuracy=accuracy_score(Y, prediction_output)
        if (1-Accuracy) < min_error:
            break
        gradient=(np.dot(X.T,((Y-y_hat) * (1-(y_hat ** 2)))))/sample_count 
        weight+=np.multiply(learning_rate,gradient)
        
        #if i%50 == 0:
           # print("at "+str(i)+": cost: "+str(L2_loss))

    #print("at "+str(i)+": cost: "+str(Accuracy))
    return weight

def get_Dataset(feature_data_frame): 
    X = feature_data_frame.drop(columns=feature_data_frame.columns[-1])
    Y = feature_data_frame[feature_data_frame.columns[-1]]
    data = pd.concat([X,Y],axis=1)
    X=X.to_numpy()
    sample,attributes=X.shape
    Y=Y.to_numpy().reshape(sample,1)
    
    dummy=np.ones((sample,1))
    X=np.concatenate((dummy,X),axis=1)
    return X,Y,data

def split(feature_data_frame):

    X = feature_data_frame.drop(columns=feature_data_frame.columns[-1])
    Y = feature_data_frame[feature_data_frame.columns[-1]]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) #42

    training_data = pd.concat([X_train,Y_train],axis=1)
    testing_data = pd.concat([X_test,Y_test],axis=1)
    #print(len(training_data.values))
    

    X_train=X_train.to_numpy()
    sample,attributes=X_train.shape
    dummy=np.ones((sample,1))
    X_train=np.concatenate((dummy,X_train),axis=1)
    Y_train=Y_train.to_numpy().reshape(sample,1)
    
    X_test=X_test.to_numpy()
    sample2,attributes2=X_test.shape
    Y_test=Y_test.to_numpy().reshape(sample2,1)
    dummy=np.ones((sample2,1))
    X_test=np.concatenate((dummy,X_test),axis=1)
    return X_train, X_test, Y_train, Y_test,training_data,testing_data

def find_prediction(Y,y_hat):
    prediction_output=(y_hat>=0)*2 - 1
    Accuracy=accuracy_score(Y, prediction_output)
    print('Accuracy Score :',round(Accuracy*100,4))
    return prediction_output

def find_performance_measure(Y,y_hat):
    eps=0.000000000001
    
    #prediction_output=(y_hat>=0)*2 - 1
    prediction_output=find_prediction(Y,y_hat)
    training_results=confusion_matrix(Y,prediction_output)
    tn,fp,fn,tp=training_results.ravel()
    
    #Accuracy=accuracy_score(Y, prediction_output)
    Recall=(tp/(tp+fn+eps)) #sensitivity,recall,hitb rate,tpr
    Precision=(tp/(tp+fp+eps))#PPV(positive predictive value)
    TNR=(tn/(tn+fp+eps)) #specifity,selectivity,tnr
    F1_score=((2*Recall*Precision)/(Recall+Precision+eps)) #harmonic mean
    FDR=1-Precision # fp/(tp+fp) ,false discovery rate
    print(tp,fn,fp,tn)
    print((tp+tn)/(tp+tn+fp+fn))
    
    
    print("True Positive Rate : ",round(Recall*100,4))
    print("True Negative Rate : ",round(TNR*100,4))
    print("Positive Predictive Value : ",round(Precision*100,4))
    print("False Discovery Rate : ",round(FDR*100,4))
    print("F1 Score : ",round(F1_score*100,4))

def resample(examples,sample_weight):
    new_dataset=[]
    indexed_dataset=examples.values
    indexes = np.random.choice(np.arange(0,len(indexed_dataset)),len(indexed_dataset),p=sample_weight)
    
    for i in indexes:
        new_dataset.append(indexed_dataset[i])
    new_dataset=pd.DataFrame(new_dataset,columns=examples.columns)
    return new_dataset

def Normalize(w_vector):
    w_vector=[x/np.sum(w_vector) for x in w_vector]
    return w_vector

def AdaBoost(k,X,Y,examples,LR):
    #X = examples.drop(columns=examples.columns[-1])
    #Y = examples[examples.columns[-1]]
    
    sample_count,column=Y.shape
    
    weight_vector=[1/sample_count for x in range(sample_count)]
    hypothesis_vector=[]
    hypothesis_weighted_vector=[0 for x in range(k)]
    for k in range(k):
        
        dataset=resample(examples,weight_vector)
        X_train, Y_train,training_data=get_Dataset(dataset)
        learned_weight=LR(X_train,Y_train,0.1,0.5,1000) #minimum error thereshold 0.5
        hypothesis_vector.append(learned_weight)
        
        
        y_hat=find_predicted_y_hat(np.dot(X,learned_weight))
        prediction_output=(y_hat>=0)*2 - 1
        
        error=0
        for j in range(sample_count):
            if prediction_output[j] != Y[j]:
                error+=weight_vector[j]
        
        if error > 0.5:
            continue
        for j in range(sample_count):
            if prediction_output[j] == Y[j]:
                weight_vector[j]=weight_vector[j]*(error/(1-error))
        weight_vector=Normalize(weight_vector)
        hypothesis_weighted_vector[k]=math.log((1-error)/error,2)
   
    hypothesis_weighted_vector=Normalize(hypothesis_weighted_vector)
   
    return hypothesis_vector,hypothesis_weighted_vector #h,z where h is the vector of learned weight and z=weight of hypothesis

def find_weighted_majority(h,z,X):
    n,m=X.shape
    weighted_sum=np.zeros((n,1))
    for i in range(len(h)):
        y_hat=find_predicted_y_hat(np.dot(X,h[i]))
        weighted_value=np.multiply(z[i],y_hat)
        weighted_sum+=weighted_value
    return weighted_sum

def model_LR(X_train,Y_train,X_test,Y_test,learning_rate,min_error,max_iter):
    weight=Logistic_Regression(X_train,Y_train,learning_rate,min_error,max_iter) #hyperparameter
    print("\n")
    print("Performance of Logistic Regression on training set: ")
    y_hat_on_trainingSet=find_predicted_y_hat(np.dot(X_train,weight))
    find_performance_measure(Y_train,y_hat_on_trainingSet)
    print("\n")
    print("Performance of Logistic Regression on test set: ")
    y_hat_on_testSet=find_predicted_y_hat(np.dot(X_test,weight))
    find_performance_measure(Y_test,y_hat_on_testSet)

def model_Adaboost(k,X_train,Y_train,X_test,Y_test,training_data):
    hypothesis_vector,hypothesis_weighted_vector=AdaBoost(k,X_train,Y_train,training_data,Logistic_Regression)
    #weighted sum kora lagbe then classify
    weighted_sum=find_weighted_majority(hypothesis_vector,hypothesis_weighted_vector,X_train)
    print("\n")
    print("Performance of Adaboost on training set: ")
    find_prediction(Y_train,weighted_sum)
    #find_performance_measure(Y_train,weighted_sum)
    weighted_sum=find_weighted_majority(hypothesis_vector,hypothesis_weighted_vector,X_test)
    print("\n")
    print("Performance of Adaboost on test set: ")
    find_prediction(Y_test,weighted_sum)

def training_on_Telco_Dataset(k,top_features):
    feature_data_frame = read_Telco_Customer_Churn_Dataset()
    dataset_of_weak_learner=read_Telco_Customer_Churn_Dataset_for_weak_learner(top_features) #feature selection
    X_train, X_test, Y_train, Y_test,training_data,testing_data=split(feature_data_frame)
    #X_train, X_test, Y_train, Y_test,training_data,testing_data=split(dataset_of_weak_learner) #feature selection
    
    model_LR(X_train,Y_train,X_test,Y_test,0.1,0,1000)

    X_train, X_test, Y_train, Y_test,training_data,testing_data=split(dataset_of_weak_learner) #feature selection
    
    model_Adaboost(k,X_train,Y_train,X_test,Y_test,training_data)

def training_on_Adult_Dataset(k,top_features):
    feature_data_frame_train= read_adult_training_dataset()
    feature_data_frame_test=read_adult_test_dataset()
    train_dataset_of_weak_learner=read_adult_training_dataset_for_weak_learner(top_features) #feature selection
    test_dataset_of_weak_learner=read_adult_test_dataset_for_weak_learner(top_features) #feature selection
    
    X_train, Y_train,training_data=get_Dataset(feature_data_frame_train)
    X_test,Y_test,testing_data=get_Dataset(feature_data_frame_test)

    model_LR(X_train,Y_train,X_test,Y_test,0.1,0,1000)

    X_train, Y_train,training_data=get_Dataset(train_dataset_of_weak_learner) #feature selection
    X_test,Y_test,testing_data=get_Dataset(test_dataset_of_weak_learner) #feature selection

    model_Adaboost(k,X_train,Y_train,X_test,Y_test,training_data)

def training_on_creditCard_Dataset(k,negSample_value):
    
    feature_data_frame = read_creditCard_dataset(negSample_value)
    #dataset_of_weak_learner=read_creditCard_Dataset_for_weak_learner(negSample_value,15)#feature selection
    X_train, X_test, Y_train, Y_test,training_data,testing_data=split(feature_data_frame)
    #X_train, X_test, Y_train, Y_test,training_data,testing_data=split(dataset_of_weak_learner) #feature selection
    model_LR(X_train,Y_train,X_test,Y_test,0.1,0,1000)

    #model_Adaboost(k,X_train,Y_train,X_test,Y_test,training_data)


training_on_Telco_Dataset(5,8) #19 features
#training_on_Adult_Dataset(5,8) #14 features
#training_on_creditCard_Dataset(5,20000) #30 features better 6000 neg samples 