import torch
import sys
sys.path.append('./build/lib.macosx-10.13-x86_64-3.6')
import main

if __name__=="__main__" :
    A = torch.zeros([10,10], dtype=torch.float32) + 5
    B = main.forward(A)[0]

    print(B)
