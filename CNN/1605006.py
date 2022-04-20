
from math import floor
import math
import time
import numpy as np
np.random.seed(6)
from keras.datasets import mnist
from keras.datasets import cifar10
#import idx2numpy
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def load_CIFAR_10_dataset():
    (training_X,training_Y),(test_X,test_Y)=cifar10.load_data()
    print(training_X.shape,training_Y.shape,test_X.shape,test_Y.shape)
    training_X=training_X/255
    test_X=test_X/255
    training_Y=training_Y.reshape(-1)
    test_Y=test_Y.reshape(-1)
    indices = np.random.permutation(test_X.shape[0])
    validation_size=int(test_X.shape[0]/2)          # permute and split testing data in  test and validation sets
    validation_idx,test_idx = indices[:validation_size], indices[validation_size:]  

    test_images= test_X[test_idx, :]
    validation_X = test_X[validation_idx, :]
    test_labels=test_Y[test_idx]
    validation_Y =  test_Y[validation_idx]
    return training_X,training_Y,validation_X,validation_Y,test_images,test_labels

def load_MNIST_dataset():
    #training_X=idx2numpy.convert_from_file("/content/train-images.idx3-ubyte")
    #training_Y=idx2numpy.convert_from_file("/content/train-labels.idx1-ubyte")#.reshape(-1,1)
    #test_X=idx2numpy.convert_from_file("/content/t10k-images.idx3-ubyte")
    #test_Y=idx2numpy.convert_from_file("/content/t10k-labels.idx1-ubyte")#.reshape(-1,1)
    (training_X,training_Y),(test_X,test_Y)=mnist.load_data()
    print(training_X.shape,training_Y.shape,test_X.shape,test_Y.shape)
    training_X=training_X/255
    training_X=np.reshape(training_X,(training_X.shape[0],training_X.shape[1],training_X.shape[2],1))
    test_X=test_X/255
    test_X=np.reshape(test_X,(test_X.shape[0],test_X.shape[1],test_X.shape[2],1))

    indices = np.random.permutation(test_X.shape[0])
    validation_size=int(test_X.shape[0]/2)          # permute and split testing data in  test and validation sets
    validation_idx,test_idx = indices[:validation_size], indices[validation_size:]  
    
    test_images= test_X[test_idx, :]
    validation_X = test_X[validation_idx, :]
    test_labels=test_Y[test_idx]
    validation_Y =  test_Y[validation_idx]
    #print(training_X.shape,training_Y.shape,test_images.shape,test_labels.shape,validation_X.shape,validation_Y.shape)
    #exit()
    return training_X,training_Y,validation_X,validation_Y,test_images,test_labels
    #print(training_Y[:10]) 
class Convolution_Layer:
    def __init__(self,number_of_filters,filter_dimension,stride,padding,number_of_channels,input_dim):
        self.number_of_filters=number_of_filters
        self.filter_dimension=filter_dimension
        self.padding=padding
        self.stride=stride
        self.number_of_channels=number_of_channels
        self.input_dim=input_dim
        #print(self.number_of_channels)
        #self.filter=np.random.uniform(-1,1,(self.filter_dimension,self.filter_dimension,self.number_of_channels,self.number_of_filters))
        self.filter=np.random.randn(self.filter_dimension,self.filter_dimension,self.number_of_channels,self.number_of_filters)*(np.sqrt(2/self.input_dim))
        self.previous_output_image=None
        #self.bias=np.zeros((self.number_of_filters,1))
        self.bias=np.zeros((1,1,1,self.number_of_filters))
    
    def forward_pass(self,previous_image):
        
        self.previous_output_image=previous_image
        #number_of_channels=previous_image.shape[-1]
        number_of_samples=previous_image.shape[0]

        #self.filter=np.random.uniform(-1,1,(self.filter_dimension,self.filter_dimension,number_of_channels,self.number_of_filters)) #not here
        
        input_height=previous_image.shape[1] # if batch of images are sent then shape 1,2 will height,width
        input_width=previous_image.shape[2]
   
        previous_image=np.pad(previous_image,((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        #previous_image=np.pad(previous_image,((self.padding,self.padding),(self.padding,self.padding),(0,0)))
        
        output_height=floor((input_height-self.filter_dimension+(2*self.padding))/self.stride)+1
        output_width=floor((input_width-self.filter_dimension+(2*self.padding))/self.stride)+1
        convolution_output=np.zeros((number_of_samples,output_height,output_width,self.number_of_filters))
        # for m in range(number_of_samples):
        #     previous_image2=previous_image[m,:]
        #     for i in range(output_height):
        #         for j in range(output_width):
        #             t1=i*self.stride
        #             t2=j*self.stride
        #             patch_image=previous_image2[t1:t1+self.filter_dimension,t2:t2+self.filter_dimension,:]
        #             for f in range(self.number_of_filters):
                            
        #                 temp=np.multiply(patch_image,self.filter[:,:,:,f])
        #                 #print(patch_image,self.filter[:,:,0,f])
        #                 convolution_output[m,i,j,f]=np.sum(temp)+self.bias[:,:,:,f]
        for i in range(output_height):
            for j in range(output_width):
                t1=i*self.stride
                t2=j*self.stride
                patch_image=previous_image[:,t1:t1+self.filter_dimension,t2:t2+self.filter_dimension,:]
                #print(patch_image.shape,self.filter.shape)
                
                for f in range(self.number_of_filters):        
                    temp=np.multiply(patch_image,self.filter[:,:,:,f])
                    #print(patch_image,self.filter[:,:,0,f])
                    convolution_output[:,i,j,f]=np.sum(temp,axis=(1,2,3))+self.bias[:,:,:,f]
                
        #print(convolution_output.shape)
        return convolution_output
    def backward_pass(self,gradient,learning_rate):
        #print(self.previous_output_image.shape)
        number_of_samples=self.previous_output_image.shape[0]
        
        output_height=gradient.shape[1]
        output_width=gradient.shape[2]
        
        
        number_of_channels=self.previous_output_image.shape[3] #if not batch
        
        dX=np.zeros(self.previous_output_image.shape)
        dW=np.zeros(self.filter.shape)
        dB=np.zeros(self.bias.shape)
        #A_prev_pad = zero_pad(A_prev, pad) #previous_output_image
        self.previous_output_image=np.pad(self.previous_output_image,((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        #dA_prev_pad = zero_pad(dA_prev, pad) #dX
        dX2=np.pad(dX,((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        # for m in range(number_of_samples):
        #     previous_image2=self.previous_output_image[m,:]
        #     dX2=dX3[m,:]
        #     for i in range(output_height):
        #         for j in range(output_width):
        #             t1=i*self.stride
        #             t2=j*self.stride
        #             for f in range(self.number_of_filters):
                        
        #                 patch_image=previous_image2[t1:t1+self.filter_dimension,t2:t2+self.filter_dimension,:]
                        
        #                 dX2[t1:t1+self.filter_dimension,t2:t2+self.filter_dimension,:]+= self.filter[:,:,:,f]*gradient[m,i,j,f]
        #                 dW[:,:,:,f]+=patch_image*gradient[m,i,j,f]
        #                 dB[:,:,:,f]+=gradient[m,i,j,f]
        
        for i in range(output_height):
            for j in range(output_width):
                t1=i*self.stride
                t2=j*self.stride
                for f in range(self.number_of_filters):
                        
                    patch_image=self.previous_output_image[:,t1:t1+self.filter_dimension,t2:t2+self.filter_dimension,:]
                    temp_filter=self.filter[:,:,:,f].reshape(1,self.filter_dimension*self.filter_dimension*self.number_of_channels)
                    

                    temp_gradient= gradient[:,i,j,f].reshape(-1,1)
                    #print(temp_gradient.shape,temp_filter.shape)
                    
                    dX2[:,t1:t1+self.filter_dimension,t2:t2+self.filter_dimension,:]+= np.dot(temp_gradient,temp_filter).reshape(-1,self.filter_dimension,self.filter_dimension,self.number_of_channels)
                    temp_gradient2=temp_gradient.T
                    
                    temp_image=patch_image.reshape(-1,self.filter_dimension*self.filter_dimension*self.number_of_channels)
                    #print(temp_gradient2.shape,temp_image.shape)
                    dW[:,:,:,f]+=np.dot(temp_gradient2,temp_image)[0].reshape(self.filter_dimension,self.filter_dimension,self.number_of_channels)
                    dB[:,:,:,f]+=np.sum(gradient[:,i,j,f])
        
            #unpadded
        if self.padding != 0:
            dX[:, :, :, :] = dX2[:,self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dX[:,:,:,:]=dX2
        self.filter-=(dW*learning_rate)
        self.bias-=(dB*learning_rate)
        
        return dX
        
class MaxPool_Layer:
    def __init__(self,filter_dimension,stride):
        self.filter_dimension=filter_dimension
        self.stride=stride
        self.previous_output_image=None
    def forward_pass(self,previous_image):
        self.previous_output_image=previous_image
        number_of_samples=previous_image.shape[0]
        input_height=previous_image.shape[1] # if batch of images are sent then shape 1,2 will height,width
        input_width=previous_image.shape[2]
        output_height=floor((input_height-self.filter_dimension)/self.stride)+1
        output_width=floor((input_width-self.filter_dimension)/self.stride)+1
        number_of_channels=previous_image.shape[3] #if not batch
        max_pooling_output=np.zeros((number_of_samples,output_height,output_width,number_of_channels))
        # for m in range(number_of_samples):
        #     previous_image2=previous_image[m,:]
        #     for i in range(output_height):
        #         for j in range(output_width):
        #             t1=i*self.stride
        #             t2=j*self.stride
        #             for f in range(number_of_channels):
        #                 patch_image=previous_image2[t1:t1+self.filter_dimension,t2:t2+self.filter_dimension,f]
        #                 max_pooling_output[m,i,j,f]=np.max(patch_image)
        for i in range(output_height):
            for j in range(output_width):
                t1=i*self.stride
                t2=j*self.stride
                
                patch_image=previous_image[:,t1:t1+self.filter_dimension,t2:t2+self.filter_dimension,:]
                max_pooling_output[:,i,j,:]=np.max(patch_image,axis=(1,2))
                
                
        #print(max_pooling_output.shape)
        return max_pooling_output
    def create_mask(self,input):
        sample,row,col=input.shape
        #input 32,2,2
        # mask=np.zeros(input.shape).reshape(-1,1)
        # mask[np.argmax(input)]=1
        # mask=mask.reshape(input.shape) 
        #mask = (input == np.max(input))
        input2=input.reshape(sample,row*col)
        max_index=np.argmax(input2,axis=1)+[j*row*col for j in range(sample)]
        mask=np.zeros(input.shape)
        mask[np.unravel_index(max_index,input.shape)]=1
        return mask
        
    def backward_pass(self,gradient,learning_rate):

        number_of_samples=self.previous_output_image.shape[0]
        
        output_height=gradient.shape[1]
        output_width=gradient.shape[2]
        number_of_channels=self.previous_output_image.shape[3] #if not batch
        
        dX=np.zeros(self.previous_output_image.shape)
        # for m in range(number_of_samples):
        #     previous_image2=self.previous_output_image[m,:]
        #     for i in range(output_height):
        #         for j in range(output_width):
        #             t1=i*self.stride
        #             t2=j*self.stride
        #             for f in range(number_of_channels):
                        
        #                 patch_image=previous_image2[t1:t1+self.filter_dimension,t2:t2+self.filter_dimension,f]
        #                 mask=self.create_mask(patch_image)
        #                 dX[m,t1:t1+self.filter_dimension,t2:t2+self.filter_dimension,f]+=mask*gradient[m,i,j,f]
        
        for i in range(output_height):
            for j in range(output_width):
                t1=i*self.stride
                t2=j*self.stride
                for f in range(number_of_channels):
                        
                    patch_image=self.previous_output_image[:,t1:t1+self.filter_dimension,t2:t2+self.filter_dimension,f]
                    mask=self.create_mask(patch_image)
                    dX[:,t1:t1+self.filter_dimension,t2:t2+self.filter_dimension,f]+=mask*gradient[:,i,j,f,np.newaxis,np.newaxis]
        
        #print(dX.shape)
        return dX
                        
class Relu_Layer:
    def __init__(self):
        self.previous_output_image=None
        
    def forward_pass(self,previous_image):
        self.previous_output_image=previous_image
        Relu_output=np.where(previous_image <= 0,0,previous_image)
        #print(Relu_output.shape)
        return Relu_output
    def backward_pass(self,gradient,learning_rate):
        Relu_derivative=np.where(self.previous_output_image <= 0,0,1)
        dX=np.zeros(self.previous_output_image.shape)
        dX=gradient * Relu_derivative
        return dX
class Softmax_Layer:
    def __init__(self) -> None:
        pass
    def forward_pass(self,previous_image):
        return np.exp(previous_image.T)/np.sum(np.exp(previous_image.T),axis=0)
    def backward_pass(self,gradient,learning_rate):
        return gradient
class Flattening_layer:
    def forward_pass(previous_image):
        sample_count=previous_image.shape[0]
        return previous_image.flatten('C').reshape(sample_count,-1)
        #return previous_image.flatten('C').reshape(-1,1)
class Fully_Connected_Layer:
    def __init__(self,output_dimension,input_dimension):
        self.output_dimension=output_dimension
        self.input_dimension=input_dimension
        self.previous_output_image=None #Last_input
        self.Last_output_image=None
        self.previous_image_shape=None #shape of before flattening 
        self.bias=np.zeros((self.output_dimension,1))
        #self.weight=np.random.randn(self.output_dimension,input_dimension)/input_dimension
        self.weight=np.random.randn(self.output_dimension,self.input_dimension)*(np.sqrt(2/self.input_dimension))

    def forward_pass(self,previous_image):
        self.previous_image_shape=previous_image.shape
        previous_image=Flattening_layer.forward_pass(previous_image)
        #print(previous_image.shape)
        input_dimension=previous_image.shape[1]
        sample_count=previous_image.shape[0]
        self.previous_output_image=previous_image
        
        #self.weight=np.random.randn(self.output_dimension,input_dimension)/input_dimension #not here

        FC_output=np.zeros((self.output_dimension,sample_count)) 
        FC_output=np.dot(self.weight,previous_image.T) + self.bias
        FC_output=FC_output.T
        self.Last_output_image=FC_output
        #print(FC_output)
        return FC_output
    def backward_pass(self,gradient,learning_rate):
        dX=np.zeros(self.previous_output_image.shape)
        dW=np.zeros(self.weight.shape)
        dB=np.zeros(self.bias.shape)
        
        dX=np.dot(gradient,self.weight)
        dW=np.dot(gradient.T,self.previous_output_image)
        dB=np.sum(gradient.T,axis=1,keepdims=True)
        
        self.weight-=(dW*learning_rate)
        self.bias-=(dB*learning_rate)
        return dX.reshape(self.previous_image_shape)

def find_cross_entropy_loss(y_predicted,y_actual):
    #print(-np.log(y_predicted)*y_actual)
    return np.sum(-np.log(y_predicted) * y_actual)
    

def read_input_file(dim,channel_count):
    FC_count=0
    layer_list=[]
    #input_file=open("input.txt","r")
    input_file=open("/content/input.txt","r")
    for line in input_file:
        #print(dim)
        a=line.split(" ")
        layer=a[0].strip()
        conv_input=dim*dim*channel_count
        if layer.lower()=="conv":
            Conv=Convolution_Layer(int(a[1]),int(a[2]),int(a[3]),int(a[4]),channel_count,conv_input)
            channel_count=int(a[1])
            dim=floor((dim-int(a[2])+(2*int(a[4])))/int(a[3]))+1
            layer_list.append(Conv)
        elif layer.lower()=="relu": 
            Relu=Relu_Layer()
            layer_list.append(Relu)
              
        elif layer.lower()=="pool":
            Pool=MaxPool_Layer(int(a[1]),int(a[2]))
            dim=floor((dim-int(a[2]))/int(a[2]))+1
            layer_list.append(Pool)
        elif layer.lower()=="fc":
            
            if FC_count==0:
                FC_count+=1
                input_dim=dim*dim*channel_count
                
            else:
                input_dim=temp
            FC=Fully_Connected_Layer(int(a[1]),input_dim)
            layer_list.append(FC)
            temp=int(a[1])
        elif layer.lower()=="softmax":
            Softmax=Softmax_Layer()
            layer_list.append(Softmax)
    input_file.close()
    return layer_list



#print(Layer_list)
def forward_passes(Image,Label,Layer_list):
    x=Image
    sample=x.shape[0]
    
    for layer in Layer_list:
        x=layer.forward_pass(x)
    y_predicted=x.T #(32,10)
    
    predicted_label=np.argmax(y_predicted,axis=1)
    #print(predicted_label)
    F1_score=f1_score(Label, predicted_label, average='macro')
    #print(F1_score)
    #one hot encode
    one_hot_Label = np.zeros((Label.shape[0], y_predicted.shape[1])) #(32,10)
    one_hot_Label[np.arange(Label.shape[0]),Label] = 1
    
    loss=find_cross_entropy_loss(y_predicted,one_hot_Label)/sample
    
    comparison=np.equal(predicted_label,Label)
    
    accuracy=np.count_nonzero(comparison)
    return y_predicted,loss,accuracy,F1_score

def backward_passes(gradient,Layer_list,learning_rate):
    for layer in reversed(Layer_list):
        gradient=layer.backward_pass(gradient,learning_rate)

def train(train_image,output_label,Layer_list,learning_rate):
    sample_count=train_image.shape[0]
    y_predicted,loss,accuracy,F1_score=forward_passes(train_image,output_label,Layer_list)
    
    one_hot_Label = np.zeros((y_predicted.shape)) #(32,10)
    one_hot_Label[np.arange(output_label.shape[0]),output_label] = 1
    gradient=np.ones((y_predicted.shape)) #initialize gradient
    gradient=(y_predicted-one_hot_Label)/sample_count
    gradient=backward_passes(gradient,Layer_list,learning_rate)
    
    return loss,accuracy

def run_validation_set(validation_X,validation_Y,Layer_list):
    #print(validation_Y.shape,validation_X.shape)
    print("Validation Set: ")
    y_predicted,L,Acc,F1_score=forward_passes(validation_X,validation_Y,Layer_list)
    Acc=(Acc/validation_X.shape[0])*100
    print("validation loss: ",L,"validation accuracy: ",Acc,"macro_F1 score: ",F1_score)

def run_test_set(test_X,test_Y,Layer_list):
    print("Test Set: ")
    y_predicted,L,Acc,F1_score=forward_passes(test_X,test_Y,Layer_list)
    Acc=(Acc/test_X.shape[0])*100
    print("test loss: ",L,"test accuracy: ",Acc,"macro_F1 score: ",F1_score)

def run_training_set(training_X,training_Y,Layer_list): 
    
    
    learning_rate=0.02
    for epoch in range(5):

        total_loss=0
        total_acc=0
        
        for i in range(int(training_Y.shape[0]/32)):
            t1=time.time()
            #print(i)
            L,Acc= train(training_X[i*32:i*32+32],training_Y[i*32:i*32+32],Layer_list,learning_rate)
            
            total_loss+=L
            total_acc+=Acc
            t2=time.time()
            #print(i,t2-t1,L,Acc)
            print("Batch count: ",i+1)
        total_loss/=((epoch+1)*(i+1))
        print("Epoch count: ",epoch+1)
        total_acc=(total_acc/training_X.shape[0])*100
        print("Training Set: \n","train loss: ",total_loss,"training accuracy: ",total_acc)
        #learning_rate*=0.1
        run_validation_set(validation_X,validation_Y,Layer_list)


training_X,training_Y,validation_X,validation_Y,test_images,test_labels=load_MNIST_dataset()
#training_X,training_Y,validation_X,validation_Y,test_images,test_labels=load_CIFAR_10_dataset()
Layer_list=read_input_file(training_X.shape[1],training_X.shape[-1])

run_training_set(training_X,training_Y,Layer_list)

run_test_set(test_images,test_labels,Layer_list)


