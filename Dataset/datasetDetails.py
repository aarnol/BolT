
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
        "dynamicLength" : 33,
        "foldCount" : 5,
        "atlas" : "sphere",
        "nOfEpochs" : 10,
        "batchSize" : 32,
        "normalize" : True,
        "fNIRS": True,

    },

    "hcpWM_fNIRS": {
        "datasetName" : "hcpfNIRS",
        "targetTask" : "nback",
        "nOfClasses" : 2,        
        "dynamicLength" : 33,
        "foldCount" : None,
        "atlas" : None,
        "nOfEpochs" : 0,
        "batchSize" : 32,
        "normalize" : True,
        "fNIRS": True,
    },


 }



