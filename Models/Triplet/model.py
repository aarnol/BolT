from Models.Triplet.bolT import BolT, ShallowClassifier
import torch
import torch.nn.functional as F
import numpy as np


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function for pairs of samples.
    """
    def __init__(self, margin=1.0, distance_function=None):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance_function = distance_function or self.euclidean_distance
    
    def euclidean_distance(self, x, y):
        return F.pairwise_distance(x, y, p=2)
    
    def cosine_distance(self, x, y):
        return 1 - F.cosine_similarity(x, y, dim=1)
    
    def forward(self, embedding1, embedding2, labels):
        """
        Args:
            embedding1: First embedding tensor [batch_size, embedding_dim]
            embedding2: Second embedding tensor [batch_size, embedding_dim]
            labels: Binary labels (1 for similar/positive pairs, 0 for dissimilar/negative pairs)
        """
        distances = self.distance_function(embedding1, embedding2)
        
        # Contrastive loss: 
        # For positive pairs (label=1): minimize distance
        # For negative pairs (label=0): maximize distance up to margin
        positive_loss = labels * torch.pow(distances, 2)
        negative_loss = (1 - labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        
        loss = 0.5 * (positive_loss + negative_loss)
        return loss.mean()


class Model():

    def __init__(self, hyperParams, details):
        self.hyperParams = hyperParams
        self.details = details

        self.model = BolT(hyperParams, details)
        self.model = self.model.to(details.device)

        # Contrastive loss (instead of triplet loss)
        self.criterion = ContrastiveLoss(
            margin=hyperParams.margin,
            distance_function=self.cosine_distance if hasattr(hyperParams, 'use_cosine') and hyperParams.use_cosine else None
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
        Step function for contrastive learning.
        
        Args:
            batch: Dictionary containing:
                - 'sample1': Dictionary with first samples
                    - 'timeseries': tensor of shape (batch_size, n_rois, time)
                    - 'label': tensor of shape (batch_size,)
                    - 'modality': tensor of shape (batch_size,)
                    - 'subjId': list of subject IDs
                - 'sample2': Dictionary with second samples (same structure as sample1)
                - 'similarities': tensor of shape (batch_size,) with 1/0 labels
            train: bool, whether in training mode
            
        Returns:
            loss: computed contrastive loss
            embeddings1: computed embeddings for first samples
            embeddings2: computed embeddings for second samples
            similarities: similarity labels
        """
        
        # Extract batch data
        sample1 = batch['sample1']
        sample2 = batch['sample2']
        similarities = batch['similarities']
        
        timeseries1 = sample1['timeseries']  # shape: (batch_size, n_rois, time)
        timeseries2 = sample2['timeseries']  # shape: (batch_size, n_rois, time)
        labels1 = sample1['label']           # shape: (batch_size,)
        labels2 = sample2['label']           # shape: (batch_size,)
        modalities1 = sample1['modality']    # shape: (batch_size,)
        modalities2 = sample2['modality']    # shape: (batch_size,)

        # Send to device
        timeseries1 = self.prepareInput(timeseries1)[0]
        timeseries2 = self.prepareInput(timeseries2)[0]
        similarities = similarities.to(self.details.device, non_blocking=True)
        labels1 = labels1.to(self.details.device, non_blocking=True)
        labels2 = labels2.to(self.details.device, non_blocking=True)
        modalities1 = modalities1.to(self.details.device, non_blocking=True)
        modalities2 = modalities2.to(self.details.device, non_blocking=True)

        # Set model mode
        self.model.train() if train else self.model.eval()

        # Forward pass - get embeddings for both samples
        embeddings1 = self.model(timeseries1)[0]  # shape: (batch_size, embedding_dim)
        embeddings2 = self.model(timeseries2)[0]  # shape: (batch_size, embedding_dim)

        # Compute contrastive loss
        loss = self.criterion(embeddings1, embeddings2, similarities)

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        loss_value = loss.detach().cpu()
        embeddings1_cpu = embeddings1.detach().cpu()
        embeddings2_cpu = embeddings2.detach().cpu()
        
        torch.cuda.empty_cache()

        return loss_value, embeddings1_cpu, embeddings2_cpu, similarities.cpu()

    def step_single(self, batch, train=False):
        """
        Step function for single samples (useful for evaluation/inference).
        
        Args:
            batch: Dictionary containing:
                - 'timeseries': tensor of shape (batch_size, n_rois, time)
                - 'labels': tensor of shape (batch_size,)
                - 'modalities': tensor of shape (batch_size,)
                - 'subj_ids': list of subject IDs
            train: bool, whether in training mode (usually False for this function)
            
        Returns:
            embeddings: computed embeddings for the batch
            labels: labels for the batch
            modalities: modalities for the batch
        """
        # Extract batch data
        timeseries = batch['timeseries']  # shape: (batch_size, n_rois, time)
        labels = batch['labels']          # shape: (batch_size,)
        modalities = batch['modalities']  # shape: (batch_size,)
        subj_ids = batch['subj_ids']      # list of subject IDs

        # Send to device
        timeseries = self.prepareInput(timeseries)[0]
        labels = labels.to(self.details.device, non_blocking=True)
        modalities = modalities.to(self.details.device, non_blocking=True)

        # Set model mode
        self.model.train() if train else self.model.eval()

        # Forward pass - get embeddings
        with torch.no_grad() if not train else torch.enable_grad():
            embeddings = self.model(timeseries)[0]  # shape: (batch_size, embedding_dim)

        embeddings_cpu = embeddings.detach().cpu()
        
        torch.cuda.empty_cache()

        return embeddings_cpu, labels.cpu(), modalities.cpu()

    def prepareInput(self, *xs):
        """Keep the same prepareInput function"""
        return [x.to(self.details.device, non_blocking=True) for x in xs]

    def evaluate_pairs(self, embeddings1, embeddings2, similarities, threshold=0.5):
        """
        Evaluate contrastive pairs using distance threshold.
        
        Args:
            embeddings1: First set of embeddings [batch_size, embedding_dim]
            embeddings2: Second set of embeddings [batch_size, embedding_dim]
            similarities: Ground truth similarities [batch_size]
            threshold: Distance threshold for classification
            
        Returns:
            accuracy: Classification accuracy
            distances: Computed distances
        """
        # Compute distances
        distances = F.pairwise_distance(embeddings1, embeddings2, p=2)
        
        # Classify based on threshold (distance < threshold means similar)
        predictions = (distances < threshold).float()
        
        # Compute accuracy
        accuracy = (predictions == similarities).float().mean()
        
        return accuracy.item(), distances.detach().cpu().numpy()

    def compute_cross_modal_similarity(self, fmri_embeddings, fnirs_embeddings, fmri_labels, fnirs_labels):
        """
        Compute cross-modal similarity matrix between fMRI and fNIRS embeddings.
        Useful for evaluating cross-modal retrieval performance.
        
        Args:
            fmri_embeddings: fMRI embeddings [n_fmri, embedding_dim]
            fnirs_embeddings: fNIRS embeddings [n_fnirs, embedding_dim]
            fmri_labels: fMRI labels [n_fmri]
            fnirs_labels: fNIRS labels [n_fnirs]
            
        Returns:
            similarity_matrix: Cross-modal similarity matrix
            retrieval_accuracy: Top-1 retrieval accuracy
        """
        # Compute pairwise cosine similarity
        fmri_norm = F.normalize(fmri_embeddings, p=2, dim=1)
        fnirs_norm = F.normalize(fnirs_embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(fmri_norm, fnirs_norm.t())
        
        # Compute retrieval accuracy (fMRI -> fNIRS)
        _, top_indices = similarity_matrix.topk(1, dim=1)
        predicted_labels = fnirs_labels[top_indices.squeeze()]
        retrieval_accuracy = (predicted_labels == fmri_labels).float().mean()
        
        return similarity_matrix.detach().cpu().numpy(), retrieval_accuracy.item()
    def getEmbeddings(self, batch):
        """
        Get embeddings for classification.
        Can work with either single samples or pairs (using first sample).
        """
        self.model.eval()  # Set model to evaluation mode
       

        # Send to device
        timeseries = batch.to(self.details.device, non_blocking=True)

        # Forward pass - get embeddings
        with torch.no_grad():
            embeddings = self.model(timeseries)[0]
        
        
        
        torch.cuda.empty_cache()
        return embeddings

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

    def step(self, batch, train=True):
        """
        Step function for classification using embeddings.
        Can work with either single samples or pairs (using first sample).
        """
        # Handle both contrastive batch format and single sample format
    
       
        timeseries = batch[0]
        
        labels = batch[1]
        
        # Send to device
        timeseries = timeseries.to(self.details.device, non_blocking=True)
        labels = labels.to(self.details.device, non_blocking=True)

        # Set model mode
        self.model.train() if train else self.model.eval()

        # Forward pass
        outputs = self.model(timeseries)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        # Compute loss
        loss = self.criterion(logits, labels)

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        loss_value = loss.detach().cpu()
        predictions = torch.softmax(logits, dim=1).detach().cpu()
        
        torch.cuda.empty_cache()

        return loss_value, predictions, labels.cpu()

    


# Example usage in your training loop:
"""
# Initialize your contrastive dataset
dataset = BalancedContrastiveDataset(datasetDetails)
dataloader = dataset.getFold(fold=0, train=True)

# Initialize model
model = Model(hyperParams, details)

# Training loop
for batch in dataloader:
    # batch contains 'sample1', 'sample2', and 'similarities'
    loss, emb1, emb2, similarities = model.step(batch, train=True)
    
    # Evaluate pair accuracy
    accuracy, distances = model.evaluate_pairs(emb1, emb2, similarities)
    
    print(f"Loss: {loss.item():.4f}, Pair Accuracy: {accuracy:.4f}")

# For evaluation with single samples:
eval_dataset = OnlineContrastiveDataset(datasetDetails)  # or create single-sample dataset
eval_dataloader = eval_dataset.getFold(fold=0, train=False)

all_embeddings = []
all_labels = []
all_modalities = []

for batch in eval_dataloader:
    # If using contrastive dataset, extract single samples for evaluation
    if 'sample1' in batch:
        single_batch = {
            'timeseries': batch['sample1']['timeseries'],
            'labels': batch['sample1']['label'],
            'modalities': batch['sample1']['modality'],
            'subj_ids': batch['sample1']['subjId']
        }
    else:
        single_batch = batch
    
    embeddings, labels, modalities = model.step_single(single_batch, train=False)
    all_embeddings.append(embeddings)
    all_labels.append(labels)
    all_modalities.append(modalities)

# Compute cross-modal similarity
fmri_mask = torch.cat(all_modalities) == 0
fnirs_mask = torch.cat(all_modalities) == 1

fmri_embeddings = torch.cat(all_embeddings)[fmri_mask]
fnirs_embeddings = torch.cat(all_embeddings)[fnirs_mask]
fmri_labels = torch.cat(all_labels)[fmri_mask]
fnirs_labels = torch.cat(all_labels)[fnirs_mask]

sim_matrix, retrieval_acc = model.compute_cross_modal_similarity(
    fmri_embeddings, fnirs_embeddings, fmri_labels, fnirs_labels
)

print(f"Cross-modal retrieval accuracy: {retrieval_acc:.4f}")
"""