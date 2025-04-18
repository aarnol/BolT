


from utils import Option

def getHyper_bolT():

    hyperDict = {

            "weightDecay" : 0,

            "lr" : 2e-4,
            "minLr" : 2e-5,
            "maxLr" : 4e-4,

            # FOR BOLT
            "nOfLayers" : 3,
            #change this based on the atlas!
            # Schaefer 400 = 400
            # AAL = 116
            # MNI = 107
            "dim" : 107,        

            "numHeads" : 36,
            "headDim" : 20,

        
            "windowSize" : 5, #changed for shorter sequences
            "shiftCoeff" : 2.0/5.0,            
            "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
            "focalRule" : "expand",

            "mlpRatio" : 1.0,
            "attentionBias" : True,
            "drop" : 0.1,
            "attnDrop" : 0.1,
            "lambdaCons" : 1,

            # extra for ablation study
            "pooling" : "cls", # ["cls", "gmp"]         
                

    }

    return Option(hyperDict)

