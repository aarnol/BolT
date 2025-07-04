
datasetDetailsDict = {

    "hcpRest" : {
        "datasetName" : "hcpRest",
        "targetTask" : "gender",
        "taskType" : "classification",
        "nOfClasses" : 2,        
        "dynamicLength" : 600,
        "foldCount" : 5,
        "atlas" : "schaefer7_400",
        "nOfEpochs" : 20,
        "batchSize" : 32       
    },

    "hcpTask" : {
        "datasetName" : "hcpTask",
        "targetTask" : "taskClassification",
        "nOfClasses" : 7,
        "dynamicLength" : 150,
        "foldCount" : 5,
        "atlas" : "schaefer7_400",
        "nOfEpochs" : 20,
        "batchSize" : 16
    },

    "abide1" : {
        "datasetName" : "abide1",
        "targetTask" : "disease",
        "nOfClasses" : 2,        
        "dynamicLength" : 60,
        "foldCount" : 10,
        "atlas" : "schaefer7_400",
        "nOfEpochs" : 20,
        "batchSize" : 32        
    },
    "hcpWM" : {
        "datasetName" : "hcpWM",
        "targetTask" : "nback",
        "nOfClasses" : 2,        
        "dynamicLength" : 20,
        "foldCount" : 5,
        "atlas" : "schaefer7_400",
        "nOfEpochs" : 10,
        "batchSize" : 32,
        "normalize" : True,
        "fNIRS": False,    
    },
    "hcpMotor" : {
        "datasetName" : "hcpMotor",
        "targetTask" : "motor",
        "nOfClasses" : 2,        
        "dynamicLength" : 15,
        "foldCount" : 5,
        "atlas" : "schaefer7_400",
        "nOfEpochs" : 10,
        "batchSize" : 32,
        "normalize" : True,
        "fNIRS": False,    
    },
    "hcpWM_AAL" : {
        "datasetName" : "hcpWM",
        "targetTask" : "nback",
        "nOfClasses" : 2,        
        "dynamicLength" : 15,
        "foldCount" : 5,
        "atlas" : "AAL",
        "nOfEpochs" : 10,
        "batchSize" : 32,
        "normalize" : True,
        "fNIRS": False        
    },
    "hcpWM_sphere" : {
        "datasetName" : "hcpWM",
        "targetTask" : "nback",
        "nOfClasses" : 2,        
        "dynamicLength" : 10,
        "foldCount" : 5,
        "atlas" : "sphere",
        "nOfEpochs" : 10,
        "batchSize" : 16,
        "normalize" : True,
        "fNIRS": False,

    },
    "hcpMotor_sphere" : {
        "datasetName" : "hcpWM",
        "targetTask" : "motor",
        "nOfClasses" : 2,        
        "dynamicLength" : 40,
        "foldCount" : 5,
        "atlas" : "sphere",
        "nOfEpochs" : 40,
        "batchSize" : 16,
        "normalize" : True,
        "fNIRS": False,

    },
    "hcpWM_fNIRS_HbR" : {
        "datasetName" : "hcpfNIRS",
        "targetTask" : "nback",
        "nOfClasses" : 2,        
        "dynamicLength" : 15,
        "foldCount" : 5,
        "atlas" : None,
        "nOfEpochs" : 10,
        "batchSize" : 16,
        "normalize" : True,
        "fNIRS": True,
        "signal": "HBR",
        "subject": None,

    },
    "hcpWM_fNIRS_HbO" : {
        "datasetName" : "hcpfNIRS",
        "targetTask" : "nback",
        "nOfClasses" : 2,        
        "dynamicLength" : 15,
        "foldCount" : 5,
        "atlas" : None,
        "nOfEpochs" : 10,
        "batchSize" : 16,
        "normalize" : True,
        "fNIRS": True,
        "signal": "HBO",
        "subject": None,

    },
    "hcpWM_fNIRS_HbC" : {
        "datasetName" : "hcpfNIRS",
        "targetTask" : "nback",
        "nOfClasses" : 2,        
        "dynamicLength" : 20,
        "foldCount" : 5,
        "atlas" : None,
        "nOfEpochs" : 100,
        "batchSize" : 16,
        "normalize" : True,
        "fNIRS": True,
        "signal": "HBC",
        "subject": None,

    },
    "hcpWM_HbC_triplet" : {
        "datasetName" : "hcpfNIRS",
        "targetTask" : "nback",
        "nOfClasses" : 2,        
        "dynamicLength" : 30,
        "foldCount" : 30,
        "atlas" : None,
        "nOfEpochs" : 30,
        "batchSize" : 32,
        "normalize" : True,
        "fNIRS": True,
        "signal": "HBC",
        "subject": None,
        "numTriplets": 10000,
        'nOfClasses': 2,


    },
    "hcpWM_sphere_triplet" : {
        "datasetName" : "hcpfNIRS",
        "targetTask" : "nback",
        "nOfClasses" : 2,        
        "dynamicLength" : 30,
        "foldCount" : 2,
        "atlas" : "sphere",
        "nOfEpochs" : 1,
        "batchSize" : 3000,
        "normalize" : True,
        "fNIRS": False,
        "numTriplets": 10000,
        "signal": "HBC",
        'nOfClasses': 2,

    },

    
 }




configurations = []

signals = ["HbT", "HbR", "HbO"]
subjects = range(1, 7)  # Subjects 1 to 6
dynamic_lengths = [30, 15]  # Variations with dynamic lengths

for subject in subjects:
    for signal in signals:
        for dynamic_length in dynamic_lengths:
            config = {
                "datasetName": "hcpfNIRS",
                "targetTask": "nback",
                "nOfClasses": 2,
                "dynamicLength": dynamic_length,
                "foldCount": None,
                "atlas": None,
                "nOfEpochs": 0,
                "batchSize": 32,
                "normalize": True,
                "fNIRS": True,
                "subject": subject,
                "signal": signal
            }
            configurations.append(config)

# add configurations to datasetDetailsDict

for config in configurations:
    datasetDetailsDict[f"hcpfNIRS_{config['subject']}_{config['signal']}_{config['dynamicLength']}"] = config





