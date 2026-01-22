import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-i',type = str, action = 'store', dest = 'names',help="the names of all haplotypeDM files to be combined, separated by comma")
parser.add_argument('-o',type = str, action = 'store', dest = 'output',help="the names of all haplotypeDM files to be combined, separated by comma")

args = parser.parse_args()

names_list = args.names.split(",")

for i in range(len(names_list)):
    hapDM = pd.read_csv(names_list[i],sep="\t")
    if i == 0:
        combined_hapDM = hapDM
    else:
        combined_hapDM = pd.concat([combined_hapDM,hapDM],axis=1)

combined_hapDM.to_csv(args.output+"_haplotypeDM.txt",sep="\t",index=False)
