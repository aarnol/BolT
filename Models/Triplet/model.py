from Models.Triplet.bolT import BolT, ShallowClassifier
import torch
import numpy as np



class Model():

    def __init__(self, hyperParams, details):
        self.hyperParams = hyperParams
        self.details = details

        self.model = BolT(hyperParams, details)
        self.model = self.model.to(details.device)

        #triplet loss
        self.criterion = torch.nn.TripletMarginWithDistanceLoss(
            distance_function=self.cosine_distance,
            margin=hyperParams.margin,
            reduction='mean'
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=hyperParams.lr, 
            weight_decay=hyperParams.weightDecay
        )

        # Scheduler
        steps_per_epoch = int(np.ceil(details.nOfTrains / details.batchSize))
        divFactor = hyperParams.maxLr / hyperParams.lr
        finalDivFactor = hyperParams.lr / hyperParams.minLr
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=hyperParams.maxLr,
            total_steps=details.nOfEpochs * steps_per_epoch + 10,
            div_factor=divFactor,
            final_div_factor=finalDivFactor,
            pct_start=0.3
        )
    def cosine_distance(self, x, y):
        return 1 - torch.nn.functional.cosine_similarity(x, y, dim=1)


    def step(self, x, train=True):
        """
        x: list of (anchor, positive, negative) tuples — typically from DataLoader
        Each of anchor, positive, negative is a tensor of shape (N, T)
        Goal: Return loss and embeddings
        """
        if train:
            # Unpack the triplets into separate tensors
            anchors =  [y[0] for y in x]
            positives = [y[1] for y in x]
            negatives = [y[2] for y in x]
        

            # Stack them into batches
            anchor = torch.stack(anchors)     # shape: (batch, N, T)
            positive = torch.stack(positives) # shape: (batch, N, T)
            negative = torch.stack(negatives) # shape: (batch, N, T)
        else:
            anchor = x[0]
            positive = x[1]
            negative = x[2]
        #print shapes sanity check
        
        # Send to device
        anchor, positive, negative = self.prepareInput(anchor, positive, negative)

        # Set model mode
        self.model.train() if train else self.model.eval()

        # Forward pass
        anchor_embed = self.model(anchor)[0]
        positive_embed = self.model(positive)[0]
        negative_embed = self.model(negative)[0]

        # Compute triplet loss
        loss = self.criterion(anchor_embed, positive_embed, negative_embed)

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        loss = loss.detach().cpu()
        torch.cuda.empty_cache()

        return loss, anchor_embed.detach().cpu(), positive_embed.detach().cpu(), negative_embed.detach().cpu()

    def prepareInput(self, *xs):
        return [x.to(self.details.device, non_blocking=True) for x in xs]


class Classifier():
    def __init__(self, hyperParams, details):
        self.hyperParams = hyperParams
        self.details = details

        self.model = ShallowClassifier(hyperParams, details)
        self.model = self.model.to(details.device)

        # CrossEntropy loss
        self.criterion = torch.nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=hyperParams.lr, 
            weight_decay=hyperParams.weightDecay
        )

        # Scheduler
        steps_per_epoch = int(np.ceil(details.nOfTrains / details.batchSize))
        divFactor = hyperParams.maxLr / hyperParams.lr
        finalDivFactor = hyperParams.lr / hyperParams.minLr
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=hyperParams.maxLr,
            total_steps=details.nOfEpochs * steps_per_epoch + 10,
            div_factor=divFactor,
            final_div_factor=finalDivFactor,
            pct_start=0.3
        )

