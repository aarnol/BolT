# TO GET DATA READY FOR THE TESTER


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, choices=["abide1", "hcpRest", "hcpWM"], default="abide1")
parser.add_argument("-a", "--atlas", type=str, choices=["schaefer7_400", "AAL", "sphere"], default="schaefer7_400")
parser.add_argument("-f", "--fnirs", type=bool, default=False)
parser.add_argument("-n", '--name', type = str)
argv = parser.parse_args()


from Dataset.Prep.prep_abide import prep_abide, prep_hcp


if(argv.dataset == "abide1"):
    prep = prep_abide
elif(argv.dataset == "hcpWM"):
    prep = prep_hcp


prep(argv.atlas, argv.name, argv.fnirs )


