


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
            #broddmann = 41
            "dim" : 20,        

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
#Original bolt
def getTransferHyper_bolT():
    """
        Hyperparameters for transfer learning with BOLT
    """
    hyperDict = {
        "weightDecay": 0,  # Add weight decay to help with generalization

        "lr": 1e-5,           # Lower learning rate for fine-tuning
        "minLr": 5e-6,
        "maxLr": 1e-4,        # Narrower range

        # FOR BOLT
        "nOfLayers": 8,       # Keep same if not changing model depth
        "dim": 85,            # Keep same based on the atlas used

        "numHeads": 36,       # Match pre-trained model architecture
        "headDim": 20,

        "windowSize": 5,
        "shiftCoeff": 0.8,
        "fringeCoeff": 2,
        "focalRule": "expand",

        "mlpRatio": 1.0,
        "attentionBias": True,
        "drop": 0.2,          # Slightly higher dropout for regularization
        "attnDrop": 0.2,      # Same for attention dropout
        "lambdaCons": 1,

        # For ablation or downstream task performance
        "pooling": "cls",     # Or "gmp" if global pooling fits better
        }
    return Option(hyperDict)