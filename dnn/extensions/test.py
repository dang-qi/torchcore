import argparse
from tests import TestRoiAlign
from tests import TestNMS
import torch

def parse_commandline():
    parser = argparse.ArgumentParser(description='Testing differet modules')
    parser.add_argument('-d','--device',help='Device to test on', required=True)
    args = parser.parse_args()
    return args

if __name__=="__main__" :
    args = parse_commandline()

    if args.device=='cuda' and not torch.cuda.is_available() :
        args.device = 'cpu'

    print( args.device )

    #tester = TestRoiAlign()
    #tester.perform_test( device=args.device )

    tester = TestNMS()
    tester.perform_test( device=args.device )
