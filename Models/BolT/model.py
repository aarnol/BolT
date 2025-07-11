from Models.BolT.bolT import BolT
import torch
import numpy as np
from einops import rearrange



class Model():

    def __init__(self, hyperParams, details):

        self.hyperParams = hyperParams
        self.details = details

        self.model = BolT(hyperParams, details)

        # load model into gpu
        
        self.model = self.model.to(details.device)

        # set criterion
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)#, weight = classWeights)
       
        # set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = hyperParams.lr, weight_decay = hyperParams.weightDecay)

        # set scheduler
        
        steps_per_epoch = int(np.ceil(details.nOfTrains / details.batchSize))         
        
        divFactor = hyperParams.maxLr / hyperParams.lr
        finalDivFactor = hyperParams.lr / hyperParams.minLr
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, hyperParams.maxLr, details.nOfEpochs * (steps_per_epoch)+10, div_factor=divFactor, final_div_factor=finalDivFactor, pct_start=0.3)
    
    def reinitialize(self, hyperParams, details):
        """
            Reinitialize the model with new hyperparameters and details for finetuning
        """
        self.hyperParams = hyperParams
        self.details = details


        # load model into gpu
        
        self.model = self.model.to(details.device)

        # set criterion
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=hyperParams.lr, weight_decay=hyperParams.weightDecay)

        #set scheduler
        steps_per_epoch = int(np.ceil(details.nOfTrains / details.batchSize))
        finalDivFactor = hyperParams.lr / hyperParams.minLr
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, hyperParams.maxLr, details.nOfEpochs * (steps_per_epoch)+10, div_factor=hyperParams.maxLr / hyperParams.lr, final_div_factor=finalDivFactor, pct_start=0.3)
        ## SET SCHEDULER TO NONE FOR TESTING
        #self.scheduler = None
    def step(self, x, y, train=True):

        """
            x = (batchSize, N, dynamicLength) 
            y = (batchSize, numberOfClasses)

        """

        # PREPARE INPUTS
    
        inputs, y = self.prepareInput(x, y)
        
        
        # DEFAULT TRAIN ROUTINE
        
        if(train):
            self.model.train()
        else:
            self.model.eval()

        yHat, cls = self.model(*inputs)
        loss = self.getLoss(yHat, y, cls)
        


        preds = yHat.argmax(1)
        probs = yHat.softmax(1)
        # Print example logits
        #check if all prediceted classes are 1
        # if(np.sum(preds.cpu().numpy() != 1) == preds.shape[0]):
        #     print("Logits:", yHat[:5].detach().cpu().numpy())
        #     print("Predicted classes:", preds[:5].cpu().numpy())
        #     print("Ground truths:", y[:5].cpu().numpy())
       
        if(train):

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if(not isinstance(self.scheduler, type(None))):
                self.scheduler.step()            

        loss = loss.detach().to("cpu")
        preds = preds.detach().to("cpu")
        probs = probs.detach().to("cpu")

        y = y.to("cpu")
        
        torch.cuda.empty_cache()
        
        return loss, preds, probs, y
        


    # HELPER FUNCTIONS HERE

    def prepareInput(self, x, y):

        """
            x = (batchSize, N, T)
            y = (batchSize, )

        """
        # to gpu now

        x = x.to(self.details.device)
        y = y.to(self.details.device)


        return (x, ), y

    def getLoss(self, yHat, y, cls):
        
        # cls.shape = (batchSize, #windows, featureDim)

        clsLoss = torch.mean(torch.square(cls - cls.mean(dim=1, keepdims=True)))

        cross_entropy_loss = self.criterion(yHat, y)

        return cross_entropy_loss #+ clsLoss * self.hyperParams.lambdaCons


    
