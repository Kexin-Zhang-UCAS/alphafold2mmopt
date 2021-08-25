from .relax import localminimizer
from .common.protein import Protein
import sys

def main():
    pdbname=sys.argv[1]
    pdboptname=pdbname.split(".")[0]+"_opt"+".pdb"
    prot = Protein(pdbname)
    out=localminimizer.relax(prot)
    print(out,file=open(pdboptname,"w"))

if __name__=="__main__":
    main()