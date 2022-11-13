import numpy as np
import tensorflow as tf
from model_multicaps import Multicaps
from Bio import SeqIO
import os
os.environ["CUDA_VISIBLE_DEVICES"]='6'

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
# tf.config.threading.set_intra_op_parallelism_threads(5)
#tf.config.threading.set_inter_op_parallelism_threads(16)
#tf.config.experimental.set_memory_growth = True
#os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_IS"


# ------------------------------------------------------------------------------#
# pars
# ------------------------------------------------------------------------------#

figure_path = 'figure'
mini_batch = 2048
num_epochs = 150
num_classes = 2
num_dims = 64
use_mini_data = False


# chrom_path = 'D:\PyCharmWorkplace\Shanghai_Normal\\all_predict\chr6\\chr6_0_17500000.fa'  ###基因组文件


def onehot(DNA_seq):

    singleDNAFragement = np.zeros([len(DNA_seq), 4])
    # print(len(DNA_seq))
    for i in range(len(DNA_seq)):
        if DNA_seq[i] == 'A':
            singleDNAFragement[i, 0] = 1
        elif DNA_seq[i] == 'T':
            singleDNAFragement[i, 1] = 1
        elif DNA_seq[i] == 'G':
            singleDNAFragement[i, 2] = 1
        elif DNA_seq[i] == 'C':
            singleDNAFragement[i, 3] = 1
        elif DNA_seq[i] == 'N':
            singleDNAFragement[i, 0] = 0
            singleDNAFragement[i, 1] = 0
            singleDNAFragement[i, 2] = 0
            singleDNAFragement[i, 3] = 0
    return singleDNAFragement

model = Multicaps().Multicaps((50,4), num_classes, num_dims, dropout_rate1=0.17631948124336827,
                                      dropout_rate2=0.14818120039967228)
model.summary()
specie_name = "Oryctolagus_cuniculusfragment"
model.load_weights("/cluster/home/jwj/Project/Bendability/Predict_Duty/model/")
os.makedirs("/cluster/home/jwj/Project/Bendability/Predict_Duty/Result/17/" + specie_name)
gen_path = "/cluster/home/jwj/Project/Bendability/Predict_Duty/speices/17/Oryctolagus_cuniculusfragment//"

for root, dirs, files in os.walk(gen_path):
    print(files)
    for file in files:
        result_path = "/cluster/home/jwj/Project/Bendability/Predict_Duty/Result/17/" + specie_name+"/" + file.split("fa")[0] + "wiggle"  ###自己命名
        resultFile = open(result_path, "w")
        test_count = -1
        test_data = []
        for chromInfo in SeqIO.parse(gen_path + file, "fasta"):

            chromID = chromInfo.name
            chromSEQ = str(chromInfo.seq).upper()
            test_count = len(chromSEQ)
            test_data = onehot(chromSEQ)
        test_data = np.array(test_data)
        print(np.shape(test_data))
        
        target_data=[]
        resultFile.write("variableStep chrom=" + chromID + " span=" + str(1) + '\n')
        count=0
        for i in range(25,test_count-25):
            target_data.append(np.array(test_data[i-25:i+25, :]))   
            if i==25:
            	continue
            if (((i-25)%70000==0)|(i==test_count-25-1)):
                print(type(target_data)) 
                target_data = np.array(target_data)
                print(target_data.shape)
                Y_pred = model.predict(target_data)
                length = len(Y_pred)
                for j in range(length):
                    count=count+1
                    resultFile.write(str(count) + "\t" + str(Y_pred[j][0]) + "\n")
                    
                target_data = []
        resultFile.close()

