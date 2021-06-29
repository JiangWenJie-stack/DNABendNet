import numpy as np

def onehot(DNA_seq):
    """

    :param DNA_seq:
    :return: a onehot matrix
    """
    singleDNAFragement = np.zeros([len(DNA_seq),4])
    for i in range(len(DNA_seq)):
        if DNA_seq[i]=='A':
            singleDNAFragement[i,0]=1
        elif DNA_seq[i]=='T':
            singleDNAFragement[i,1]=1
        elif DNA_seq[i]=='G':
            singleDNAFragement[i,2]=1
        elif DNA_seq[i]=='C':
            singleDNAFragement[i,3]=1
        elif DNA_seq[i]=='N':
            singleDNAFragement[i, 0] = 0.25
            singleDNAFragement[i, 1] = 0.25
            singleDNAFragement[i, 2] = 0.25
            singleDNAFragement[i, 3] = 0.25
    return singleDNAFragement

def extraData(train_dir, vaild_dir, test_dir):
    #####################
    #   输入： 三类数据的地址
    #   输出：三类数据对应的数组
    #####################
    train_data = []
    train_label = []
    vaild_data = []
    vaild_label = []
    test_data = []
    test_label = []

    train_count = 0
    vaild_count = 0
    test_count = 0

    for line in open(train_dir,"r"):
        seq = line.split("\t")[0]
        label = line.split("\t")[1]
        train_count += 1
        SingeDnaMateix = onehot(seq)
        train_data.append(SingeDnaMateix)
        train_label.append(float(label))

    for line in open(vaild_dir,"r"):
        seq = line.split("\t")[0]
        label = line.split("\t")[1]
        vaild_count += 1
        SingeDnaMateix = onehot(seq)
        vaild_data.append(SingeDnaMateix)
        vaild_label.append(float(label))

    for line in open(test_dir,"r"):
        seq = line.split("\t")[0]
        label = line.split("\t")[1]
        test_count += 1
        SingeDnaMateix = onehot(seq)
        test_data.append(SingeDnaMateix)
        test_label.append(float(label))

    train_data = np.array(train_data).reshape((train_count,50,4,1))
    train_label = np.array(train_label)
    vaild_data = np.array(vaild_data).reshape((vaild_count,50,4,1))
    vaild_label = np.array(vaild_label)
    test_data = np.array(test_data).reshape((test_count,50,4,1))
    test_label = np.array(test_label)

    return train_data, train_label, vaild_data, vaild_label, test_data, test_label


