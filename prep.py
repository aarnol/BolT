# TO GET DATA READY FOR THE TESTER


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, choices=["abide1", "hcpRest", "hcpWM", "hcpMotor"], default="abide1")
parser.add_argument("-a", "--atlas", type=str, choices=["schaefer7_400", "AAL", "sphere", "brodmann"], default="schaefer7_400")
parser.add_argument("-f", "--fnirs", type=bool, default=False)
parser.add_argument("-n", '--name', type = str)
parser.add_argument("-r", "--radius", type= int, default = 30)
parser.add_argument("-s", "--smoothing", type= int, default = None)
parser.add_argument("-u", "--unique_parcels", type= bool, default = False)
argv = parser.parse_args()


from Dataset.Prep.prep_abide import prep_abide, prep_hcp


if(argv.dataset == "abide1"):
    prep = prep_abide
elif(argv.dataset == "hcpWM" or argv.dataset == "hcpMotor"):
    prep = prep_hcp


prep(argv.atlas, argv.name, argv.dataset, argv.fnirs, argv.radius, argv.smoothing, argv.unique_parcels)


