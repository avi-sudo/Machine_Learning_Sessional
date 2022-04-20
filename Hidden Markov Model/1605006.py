import numpy as np
import math
from statistics import NormalDist
np.random.seed(1605006)

def random_initialization(transition_matrix,mean_array,SD_array):
    number_of_hidden_state=len(mean_array)
    for i in range(number_of_hidden_states):
        for j in range(number_of_hidden_states):
            transition_matrix[i][j]=np.random.uniform(0,1)
    normalized_sum=np.sum(transition_matrix,axis=1)

    for i in range(number_of_hidden_states):
        for j in range(number_of_hidden_states):
            transition_matrix[i][j]/=normalized_sum[i]
    
    for i in range(number_of_hidden_states):
        mean_array[i]=np.random.randint(1,200)
        SD_array[i]=np.random.randint(1,200)

    return transition_matrix,mean_array,SD_array

    
def find_stationary_distributrion(number_of_hidden_states,transition_matrix):
    P=np.zeros((number_of_hidden_states,number_of_hidden_states))
    for i in range(number_of_hidden_states-1):
        for j in range(number_of_hidden_states):
            if i==j:
                P[i][j]=transition_matrix[j][i]-1
            else:
                P[i][j]=transition_matrix[j][i]
    for i in range(number_of_hidden_states):
        P[number_of_hidden_states-1][i]=1

    C=np.zeros(number_of_hidden_states)
    for i in range(number_of_hidden_states-1):
        C[i]=0
    C[number_of_hidden_states-1]=1

    x=np.linalg.solve(P,C)
    return x

def Viterbi_algorithm(number_of_hidden_states,number_of_observations,observations,Hidden_States,transition_matrix,mean_array,standard_deviation_array):
    #find stationary/initial/prior probability
    x=find_stationary_distributrion(number_of_hidden_states,transition_matrix)

    Max_Probability=np.zeros((number_of_hidden_states,number_of_observations+1))
    backtrack=np.zeros((number_of_hidden_states,number_of_observations+1))
    #set initial probability of hidden states
    for i in range(number_of_hidden_states):
        pdf=NormalDist(mean_array[i],standard_deviation_array[i]).pdf(observations[1])
        Max_Probability[i][1]=np.log(x[i]*pdf)

    for i in range(2,number_of_observations+1):
        for j in range(number_of_hidden_states):
            maximum=-999999999999
            previous_state=-1
            for k in range(number_of_hidden_states): 
                prob=Max_Probability[k][i-1] + np.log(transition_matrix[k][j] * NormalDist(mean_array[j],standard_deviation_array[j]).pdf(observations[i]))
                if prob > maximum:
                    maximum=prob
                    previous_state=k

            Max_Probability[j][i]=maximum
            backtrack[j][i]=previous_state

    probable_best_path=list()  
    maximum=-99999999
    final_state=-1
    for k in range(number_of_hidden_states): 
        prob=Max_Probability[k][number_of_observations]
        if prob > maximum:
            maximum=prob
            final_state=k
    #backtracking
    for i in range(number_of_observations,0,-1):
        probable_best_path.insert(0,Hidden_States[final_state])
        final_state=int(backtrack[final_state][i])

    return probable_best_path


#baulm welch implementation
def baulm_welch_algorithm(number_of_hidden_states,number_of_observations,observations,Hidden_States,transition_matrix,mean_array,standard_deviation_array):

    updated_transition_matrix=np.zeros((number_of_hidden_states,number_of_hidden_states))
    updated_mean=np.zeros((number_of_hidden_states))
    updated_SD=np.zeros((number_of_hidden_states))
    updated_variance=np.zeros((number_of_hidden_states))
    #initialization
    updated_transition_matrix=transition_matrix
    updated_mean=mean_array
    updated_SD=standard_deviation_array
    #random_initialization
    #updated_transition_matrix,updated_mean,updated_SD=random_initialization(updated_transition_matrix,updated_mean,updated_SD)
    for q in range(100):
        old_transition_matrix=updated_transition_matrix
        old_mean=updated_mean
        old_SD=updated_SD
        #forward_matrix calculation
        forward_matrix=np.zeros((number_of_hidden_states,number_of_observations+1))
        x=find_stationary_distributrion(number_of_hidden_states,updated_transition_matrix)

        sum=0
        for i in range(number_of_hidden_states):
                pdf=NormalDist(updated_mean[i],updated_SD[i]).pdf(observations[1])
                forward_matrix[i][1]=(x[i]*pdf)
                sum+=forward_matrix[i][1]
        for i in range(number_of_hidden_states):
            forward_matrix[i][1]=forward_matrix[i][1]/sum

        for i in range(2,number_of_observations+1):
            normalized_sum=0
            for j in range(number_of_hidden_states):
                summation=0
                for k in range(number_of_hidden_states):
                    summation+=forward_matrix[k][i-1]*updated_transition_matrix[k][j]*NormalDist(updated_mean[j],updated_SD[j]).pdf(observations[i])
                forward_matrix[j][i]=summation
                normalized_sum+=forward_matrix[j][i]
            for m in range(number_of_hidden_states):
                forward_matrix[m][i]=forward_matrix[m][i]/normalized_sum


        forward_sink=0
        for i in range(number_of_hidden_states):
            forward_sink+=forward_matrix[i][number_of_observations]


        #backward_matrix calculation
        backward_matrix=np.zeros((number_of_hidden_states,number_of_observations+1))

        sum=0
        for i in range(number_of_hidden_states):  
                backward_matrix[i][number_of_observations]=1
                sum+=backward_matrix[i][number_of_observations]
        for i in range(number_of_hidden_states):
            backward_matrix[i][number_of_observations]=backward_matrix[i][number_of_observations]/sum

        for i in range(number_of_observations-1,0,-1):
            normalized_sum=0
            for j in range(number_of_hidden_states):
                summation=0
                for k in range(number_of_hidden_states):
                    summation+=backward_matrix[k][i+1]*updated_transition_matrix[j][k]*NormalDist(updated_mean[k],updated_SD[k]).pdf(observations[i+1])
                backward_matrix[j][i]=summation
                normalized_sum+=backward_matrix[j][i]
            for m in range(number_of_hidden_states):
                backward_matrix[m][i]=backward_matrix[m][i]/normalized_sum

        #responsible matrix
        responsible_matrix_1=np.zeros((number_of_hidden_states,number_of_observations+1))
        for i in range(1,number_of_observations+1):
            normalized_sum=0
            for j in range(number_of_hidden_states):
                responsible_matrix_1[j][i]=(forward_matrix[j][i]*backward_matrix[j][i])/forward_sink
                normalized_sum+=responsible_matrix_1[j][i]
            for k in range(number_of_hidden_states):
                responsible_matrix_1[k][i]=responsible_matrix_1[k][i]/normalized_sum

        responsible_matrix_2=np.zeros((number_of_hidden_states*number_of_hidden_states,number_of_observations))
        for i in range(number_of_hidden_states*number_of_hidden_states):
            responsible_matrix_2[i][0]=1
            
        index=-1
        for i in range(number_of_hidden_states):
            for j in range(number_of_hidden_states):
                index+=1
                for k in range(1,number_of_observations):
                    responsible_matrix_2[index][k]=(forward_matrix[i][k]*updated_transition_matrix[i][j]*NormalDist(updated_mean[j],updated_SD[j]).pdf(observations[k+1])*backward_matrix[j][k+1])/forward_sink

        #normalize responsible matrix
        normalized_sum=np.sum(responsible_matrix_2,axis=0)
        responsible_matrix_2=np.multiply(responsible_matrix_2,1/normalized_sum)
        for i in range(number_of_hidden_states*number_of_hidden_states):
            responsible_matrix_2[i][0]=0

        transition_sum=np.sum(responsible_matrix_2,axis=1)

        idx=-1
        for i in range(number_of_hidden_states):
            for j in range(number_of_hidden_states):
                idx+=1
                updated_transition_matrix[i][j]=transition_sum[idx]
        normalized_sum=np.sum(updated_transition_matrix,axis=1)

        for i in range(number_of_hidden_states):
            for j in range(number_of_hidden_states):
                updated_transition_matrix[i][j]/=normalized_sum[i]
        
        #updated_mean=np.sum(np.multiply(responsible_matrix_1,observations),axis=1)/np.sum(responsible_matrix_1,axis=1)
        avg_sum=np.sum(responsible_matrix_1,axis=1)
        
        sum=np.sum(np.multiply(responsible_matrix_1,observations),axis=1)
        #updated_mean=sum/(avg_sum+0.000000001)
        updated_mean=sum/(avg_sum)

        for i in range(number_of_hidden_states):
            sum=0
            for j in range(1,number_of_observations+1):
                temp=(observations[j]-updated_mean[i]) ** 2
                sum+=responsible_matrix_1[i][j]*temp
                
            updated_variance[i]=sum/avg_sum[i]
            updated_SD[i]=math.sqrt(updated_variance[i])
        
        errror_sum=np.abs(np.subtract(updated_mean,old_mean))+np.abs(np.subtract(updated_SD,old_SD))+np.abs(np.subtract(updated_transition_matrix,old_transition_matrix))
        total_error=np.sum(errror_sum)
        if total_error<0.00001:
            print("Iteration: "+str(q))
            break
    return updated_transition_matrix,updated_mean,updated_SD,updated_variance


#call viterbi
def call_viterbi(number_of_hidden_states,number_of_observations,observations,Hidden_States,transition_matrix,mean_array,standard_deviation_array):
    probable_best_path=Viterbi_algorithm(number_of_hidden_states,number_of_observations,observations,Hidden_States,transition_matrix,mean_array,standard_deviation_array)

    output_file=open("1605006_states_Viterbi_wo_learning.txt","w")

    for i in probable_best_path:
        output_file.write("\""+i+"\"\n")

    output_file.close()

#call baulm_welch  
def call_Baulm_welch(number_of_hidden_states,number_of_observations,observations,Hidden_States,transition_matrix,mean_array,standard_deviation_array):     
    updated_transition_matrix,updated_mean,updated_SD,updated_variance=baulm_welch_algorithm(number_of_hidden_states,number_of_observations,observations,Hidden_States,transition_matrix,mean_array,standard_deviation_array)
    updated_stationary_probability=find_stationary_distributrion(number_of_hidden_states,updated_transition_matrix)

    output_file=open("1605006_parameters_learned.txt","w")
    output_file.write(str(number_of_hidden_states)+"\n")
    for i in range(number_of_hidden_states):
        for j in range(number_of_hidden_states):
            output_file.write(str(round(updated_transition_matrix[i][j],7))+" ")
        output_file.write("\n")
    for i in range(number_of_hidden_states):    
        output_file.write(str(round(updated_mean[i],4))+" ")
    output_file.write("\n")
    for i in range(number_of_hidden_states):    
        output_file.write(str(round(updated_variance[i],6))+" ")
    output_file.write("\n")
    for i in range(number_of_hidden_states):    
        output_file.write(str(round(updated_stationary_probability[i],2))+" ")
    output_file.close()

    probable_best_path=Viterbi_algorithm(number_of_hidden_states,number_of_observations,observations,Hidden_States,updated_transition_matrix,updated_mean,updated_SD)

    output_file=open("1605006_states_Viterbi_after_learning.txt","w")

    for i in probable_best_path:
        output_file.write("\""+i+"\"\n")

    output_file.close()

#reading input files
temp=[]
data_file=open("data.txt","r")
#data_file=open("a.txt","r")
for count,line in enumerate(data_file):
    temp.append(float(line))

number_of_observations=count+1
observations=np.zeros((number_of_observations+1))
for i in range(1,number_of_observations+1):
    observations[i]=temp[i-1]
data_file.close()

parameter_file=open("parameters.txt.txt","r")
number_of_hidden_states=int(parameter_file.readline())
transition_matrix=np.zeros((number_of_hidden_states,number_of_hidden_states))
for i in range(number_of_hidden_states):
    new_list=parameter_file.readline().split()
    for j in range(number_of_hidden_states):
        transition_matrix[i][j]=float(new_list[j])
mean_list=parameter_file.readline().split()
SD_list=parameter_file.readline().split()
parameter_file.close()
mean_array=np.zeros((number_of_hidden_states))
for i in range(number_of_hidden_states):
    mean_array[i]=float(mean_list[i])
standard_deviation_array=np.zeros((number_of_hidden_states))
for i in range(number_of_hidden_states):
    standard_deviation_array[i]=math.sqrt(float(SD_list[i]))


#state 0: El Ni˜no;;;;;;state 1:La Ni˜na
Hidden_States=["El Nino","La Nina"]

call_viterbi(number_of_hidden_states,number_of_observations,observations,Hidden_States,transition_matrix,mean_array,standard_deviation_array)
call_Baulm_welch(number_of_hidden_states,number_of_observations,observations,Hidden_States,transition_matrix,mean_array,standard_deviation_array)