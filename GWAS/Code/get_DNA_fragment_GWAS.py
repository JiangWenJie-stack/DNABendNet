import numpy as np
import pandas as pd
from Bio import SeqIO
import os

chrom_list=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y']

genomic = {}
for chromInfo in SeqIO.parse("D:\PycharmWorkspace\Cooperation\Hucongcong\Human All genomic\hg38\\hg38.fa", "fasta"):
    chromSEQ = str(chromInfo.seq).upper()
    cheomID = chromInfo.id
    if cheomID[3:] not in chrom_list:
        continue
    genomic[cheomID] = chromSEQ
    print(len(chromSEQ))


flag=0
count=0
for root, dirs, files in os.walk("D:\PycharmWorkspace\Cooperation\Hucongcong\GWAS\\", topdown=False):
    for file in files:
        fp = open("D:\PycharmWorkspace\Cooperation\Hucongcong\GWAS\\Gwas_in_snp_new"+".txt","w")
        fp1 = open("D:\PycharmWorkspace\Cooperation\Hucongcong\GWAS\\Gwas_in_snp_ref" + ".txt", "w")
        for line in open("F:\Genebentness\GWAS\\Gwas_in_snp.txt"):
            target_fragment=''
            target_chrom = line.split("\t")[2]
            if target_chrom[3:] not in chrom_list:
                #print(target_chrom)
                continue
            count+=1
            chromSEQ = genomic[target_chrom]
            target_start = int(line.split("\t")[3])
            target_end = int(line.split("\t")[4])
            target_distance = target_end-target_start
            target_shift = (-target_start+target_end)//2
            source_base = line.split("\t")[9]
            convert_base = line.split("\t")[10].split("/")

            target_style = line.split("\t")[12]
            if '-' not in line.split("\t")[0]:
                continue
            target_sp_fragment = line.split("\t")[0].split("-")[1]

            if (((target_sp_fragment ==chromSEQ[target_start:target_start+len(target_sp_fragment)]))&(((target_style=='single')|(target_style=='mnp')))):#)):
                #print(line,end='')
                continue

            #break




            if target_style =="deletion":
                prev_fragment = chromSEQ[target_start-200:target_start]
                next_fragment = chromSEQ[target_start:target_start+200]
                next_fragment = next_fragment[target_distance:]
                #print(len(next_fragment))
                all_fragment = prev_fragment+next_fragment
                target_fragment = all_fragment[200-75:200+75]
                fp.write(str(target_chrom) + "\t" + str(target_start) + "\t" + str(target_end) + "\t" + target_style + "\t"+line.split("\t")[0]+"\t"+line.split("\t")[8]+"\t")
                fp.write(target_fragment+"\n")
                fp1.write(str(target_chrom) + "\t" + str(target_start) + "\t" + str(target_end) + "\t" + target_style + "\t" + line.split("\t")[0] + "\t" + line.split("\t")[8] + "\t" + chromSEQ[target_start-1 - 75:target_start-1 + 75] + "\n")

            if target_style =="insertion":
                prev_fragment = chromSEQ[target_start-200:target_start]
                next_fragment = chromSEQ[target_start:target_start+200]

                if target_sp_fragment == '?':
                    continue
                for num in convert_base:
                    if num =="-":
                        continue
                    pp = num + next_fragment
                    all_fragment = prev_fragment+pp
                    target_fragment = all_fragment[125:275]
                    fp.write(str(target_chrom) + "\t" + str(target_start) + "\t" + str(target_end) + "\t" + target_style + "\t"+line.split("\t")[0]+"\t"+str(num)+"\t")
                    fp.write(target_fragment+"\n")
                    fp1.write(str(target_chrom) + "\t" + str(target_start) + "\t" + str(target_end) + "\t" + target_style + "\t" + line.split("\t")[0] + "\t" + line.split("\t")[8] + "\t" + chromSEQ[target_start-1 - 75:target_start-1 + 75] + "\n")

            if target_style =="in-del":
                continue

            if (target_style =="single")|(target_style=="mnp"):
                prev_fragment = chromSEQ[target_start-200:target_start]
                next_fragment = chromSEQ[target_start:target_start+200-1]
                next_fragment = next_fragment[len(source_base):]


                if target_sp_fragment == '?':
                    continue
                pp = target_sp_fragment + next_fragment
                all_fragment = prev_fragment+pp
                target_fragment = all_fragment[200+-75:200+75]
                fp.write(str(target_chrom) + "\t" + str(target_start) + "\t" + str(target_end) + "\t" + target_style + "\t"+line.split("\t")[0]+"\t"+str(target_sp_fragment)+"\t")
                fp.write(target_fragment+"\n")
                fp1.write(str(target_chrom) + "\t" + str(target_start) + "\t" + str(target_end) + "\t" + target_style + "\t" + line.split("\t")[0] + "\t" + line.split("\t")[8] + "\t" + chromSEQ[target_start- 75:target_start+ 75] + "\n")

        fp.close()
        fp1.close()
        break
    break














