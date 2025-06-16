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



    def step(self, batch, train=True):
        """
        Adapted step function for online triplet mining.
        
        Args:
            batch: Dictionary containing:
                - 'timeseries': tensor of shape (batch_size, n_rois, time)
                - 'labels': tensor of shape (batch_size,)
                - 'modalities': tensor of shape (batch_size,)
                - 'subj_ids': list of subject IDs
            train: bool, whether in training mode
            
        Returns:
            loss: computed triplet loss
            embeddings: computed embeddings for the batch
            labels: labels for the batch
            modalities: modalities for the batch
        """
        
        if train:
            # Extract batch data
            timeseries = batch['timeseries']  # shape: (batch_size, n_rois, time)
            labels = batch['labels']          # shape: (batch_size,)
            modalities = batch['modalities']  # shape: (batch_size,)
            subj_ids = batch['subj_ids']      # list of subject IDs
        else:
            # For evaluation, we might still receive the same format
            # or handle single samples differently based on your needs
            if isinstance(batch, dict):
                timeseries = batch['timeseries']
                labels = batch['labels']
                modalities = batch['modalities']
                subj_ids = batch['subj_ids']
               
            else:
                # Fallback for backward compatibility with old triplet format
                anchor, positive, negative = batch
                
                return self._step_triplet_format(anchor, positive, negative, train)

        # Send to device
        timeseries = self.prepareInput(timeseries)[0]
        labels = labels.to(self.details.device, non_blocking=True)
        modalities = modalities.to(self.details.device, non_blocking=True)

        # Set model mode
        self.model.train() if train else self.model.eval()

        # Forward pass - get embeddings for all samples
        embeddings = self.model(timeseries)[0]  # shape: (batch_size, embedding_dim)

        # Compute triplet loss using online mining
        if train:
            # Use hard triplet mining during training
            loss = self.batch_hard_triplet_loss(embeddings, labels, modalities, margin=1.0)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        else:
            # For evaluation, you might want to use all-triplet loss or just return embeddings
            loss = self.batch_all_triplet_loss(embeddings, labels, modalities, margin=1.0)

        loss_value = loss.detach().cpu()
        embeddings_cpu = embeddings.detach().cpu()
        
        torch.cuda.empty_cache()

        return loss_value, embeddings_cpu, labels.cpu(), modalities.cpu()

    def _step_triplet_format(self, anchor, positive, negative, train=True):
        """
        Backward compatibility function for old triplet format.
        This handles the case where you still have pre-formed triplets.
        """
        if train:
            # Unpack the triplets into separate tensors
            if isinstance(anchor, list):
                anchors = [y[0] for y in anchor]
                positives = [y[1] for y in anchor]
                negatives = [y[2] for y in anchor]
                
                # Stack them into batches
                anchor = torch.stack(anchors)     # shape: (batch, N, T)
                positive = torch.stack(positives) # shape: (batch, N, T)
                negative = torch.stack(negatives) # shape: (batch, N, T)

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

        loss_value = loss.detach().cpu()
        torch.cuda.empty_cache()

        return loss_value, anchor_embed.detach().cpu(), positive_embed.detach().cpu(), negative_embed.detach().cpu()

    def prepareInput(self, *xs):
        """Keep the same prepareInput function"""
        return [x.to(self.details.device, non_blocking=True) for x in xs]

    def get_triplet_mask(self, labels, modalities):
        """
        Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
        A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
        - modalities[i] != modalities[j] (cross-modal positive)
        """
        # Check that i, j and k are distinct
        indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)
        
        distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)
        
        valid_labels = i_equal_j & (~i_equal_k)
        
        # Check if modalities[i] != modalities[j] (cross-modal positive)
        modality_equal = modalities.unsqueeze(0) == modalities.unsqueeze(1)
        i_not_equal_j_modality = (~modality_equal).unsqueeze(2)
        
        return distinct_indices & valid_labels & i_not_equal_j_modality

    def batch_all_triplet_loss(self, embeddings, labels, modalities, margin=1.0):
        """
        Build the triplet loss over a batch of embeddings using online mining.
        Considers all valid triplets in the batch.
        """
        # Get the pairwise distance matrix
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
        
        # Get anchor-positive distances and anchor-negative distances
        anchor_positive_dist = pairwise_dist.unsqueeze(2)  # [batch, batch, 1]
        anchor_negative_dist = pairwise_dist.unsqueeze(1)  # [batch, 1, batch]
        
        # Compute triplet loss: max(d(a,p) - d(a,n) + margin, 0)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
        
        # Put to zero the invalid triplets
        mask = self.get_triplet_mask(labels, modalities)
        triplet_loss = mask.float() * triplet_loss
        
        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = torch.clamp(triplet_loss, min=0.0)
        
        # Count number of positive triplets (where loss > 0)
        valid_triplets = triplet_loss > 1e-16
        num_positive_triplets = valid_triplets.sum()
        
        # Get final mean triplet loss over the positive valid triplets
        if num_positive_triplets > 0:
            triplet_loss = triplet_loss.sum() / num_positive_triplets.float()
        else:
            triplet_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return triplet_loss

    def batch_hard_triplet_loss(self, embeddings, labels, modalities, margin=1.0):
        """
        Build the triplet loss over a batch of embeddings using hard mining.
        For each anchor, we select the hardest positive and hardest negative.
        """
        # Get the pairwise distance matrix
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
        
        # For each anchor, get the hardest positive
        # First, create mask for valid positives (same label, different modality)
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        modality_different = modalities.unsqueeze(0) != modalities.unsqueeze(1)
        valid_positives = label_equal & modality_different
        
        # Set distance to negatives and self to a large value so they won't be selected as hardest positive
        masked_pos_dist = pairwise_dist.clone()
        masked_pos_dist[~valid_positives] = float('inf')
        
        # Get hardest positive for each anchor
        hardest_positive_dist, _ = masked_pos_dist.min(dim=1)
        
        # For each anchor, get the hardest negative
        # First, create mask for valid negatives (different label)
        valid_negatives = ~label_equal
        
        # Set distance to positives and self to a small value so they won't be selected as hardest negative
        masked_neg_dist = pairwise_dist.clone()
        masked_neg_dist[~valid_negatives] = 0.0
        
        # Get hardest negative for each anchor
        hardest_negative_dist, _ = masked_neg_dist.max(dim=1)
        
        # Compute triplet loss for each valid anchor
        triplet_loss = hardest_positive_dist - hardest_negative_dist + margin
        triplet_loss = torch.clamp(triplet_loss, min=0.0)
        
        # Only consider anchors that have valid positives
        valid_anchors = (hardest_positive_dist != float('inf'))
        
        if valid_anchors.sum() > 0:
            triplet_loss = triplet_loss[valid_anchors].mean()
        else:
            triplet_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return triplet_loss

    # Example usage in your training loop:
    """
    # Initialize your online triplet dataset
    dataset = OnlineTripletDataset(datasetDetails)
    dataloader = dataset.getFold(fold=0, train=True)

    # Training loop
    for batch in dataloader:
        # batch is a dictionary with 'timeseries', 'labels', 'modalities', 'subj_ids'
        loss, embeddings, labels, modalities = model.step(batch, train=True)
        
        # Your logging/monitoring code here
        print(f"Loss: {loss.item()}")
"""


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

