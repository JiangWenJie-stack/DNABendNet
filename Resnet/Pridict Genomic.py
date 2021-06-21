import numpy as np
from keras.models import load_model
from Bio import SeqIO
from load2data import onehot

def Pred_Genomic(model_path, chrom_path,result_path,step):
    """

    :param model_path:
    :param chrom_path:
    :param result_path:
    :return:
    """
    model = load_model(model_path.encode('utf-8'))
    resultFile = open(result_path,"w")

    for chromInfo in SeqIO.parse(chrom_path, "fasta"):
        chromID = chromInfo.name
        chromSEQ = str(chromInfo.seq).upper()
        resultFile.write("variableStep chrom=" + str(chromID) + " span=" + str(step) + '\n')
        Pred_Matrix = onehot(chromSEQ)
       
        Pred_Matrix = np.vstack(Pred_Matrix)
        for i in range(25, len(Pred_Matrix)-25, step):
            BaseGroup_Matrix = Pred_Matrix[i-25:i+25,].reshape(1,50,4,1)
            BaseGroup_Result = model.predict(BaseGroup_Matrix)
            resultFile.write(str(i) +'\t'+ str(BaseGroup_Result[0][0]) +'\n')
    resultFile.close()

if __name__ == '__main__':
    model_path = 'D:\PycharmWorkspace\Cooperation\Hucongcong\genomic predict\model\\resnet.h5'###载入的模型
    chrom_path = 'D:\PycharmWorkspace\Cooperation\Hucongcong\genomic predict\Single genomic\chr17.fa'###基因组文件
    result_path = 'D:\PycharmWorkspace\Cooperation\Hucongcong\genomic predict\Result\\chr17.txt' ###自己命名
    Pred_Genomic(model_path=model_path,
                 chrom_path=chrom_path,
                 result_path=result_path,
                 step=7)











