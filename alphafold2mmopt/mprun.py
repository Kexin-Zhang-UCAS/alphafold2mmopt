from .relax import localminimizer
from .common.protein import Protein
import multiprocessing as mp

def RelaxOne(pdbname):
    pdboptname=pdbname.split(".")[0]+"_opt"+".pdb"
    prot = Protein(pdbname)
    out=localminimizer.relax(prot)
    print(out,file=open(pdboptname,"w"))

def Relax(pdbnames):
    procs=[]
    for name in pdbnames:
        p=mp.Process(target=RelaxOne,args=(name,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("%d tasks End!"%len(procs))

if __name__=="__main__":
    pass
