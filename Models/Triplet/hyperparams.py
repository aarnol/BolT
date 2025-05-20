


from utils import Option

def getHyper_triplet():

    hyperDict = {

            "weightDecay" : 0,

            "lr" : 2e-4,
            "minLr" : 2e-5,
            "maxLr" : 4e-4,

            # FOR BOLT
            "nOfLayers" : 8,
            #change this based on the atlas!
            # Schaefer 400 = 400
            # AAL = 116
            # MNI = 107
            "dim" : 85,        
            "margin" : .5, # margin for triplet loss
            "numHeads" : 36,
            "headDim" : 20,

        
            "windowSize" : 5, #changed for shorter sequences
            "shiftCoeff" : 0.8,            
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
