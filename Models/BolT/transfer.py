# Debugging: Loss Not Changing During Transfer Learning
import torch
def debug_loss_not_changing(model, dataLoader):
    """
    Comprehensive debugging for when loss doesn't change
    """
    print("=== DEBUGGING: Loss Not Changing ===")
    
    # 1. Check if any parameters are actually trainable
    trainable_params = []
    total_params = 0
    model = model.model if hasattr(model, 'model') else model
    for name, param in model.named_parameters():
        total_params += 1
        if param.requires_grad:
            trainable_params.append(name)
    
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {len(trainable_params)}")
    
    if len(trainable_params) == 0:
        print("❌ PROBLEM: NO TRAINABLE PARAMETERS!")
        return
    
    print("✓ Trainable parameters found:")
    for name in trainable_params:
        print(f"  - {name}")
    
    # 2. Check if gradients are flowing
    print("\n2. Checking gradient flow...")
    
    # Get a batch and do forward/backward pass
    data_iter = iter(dataLoader)
    batch = next(data_iter)
    
    xTrain = batch[0]
    yTrain = batch[1]
    
    # Manual forward pass to check gradients
    model.train()
    
    # Prepare inputs (copy from your model.py)
    x = xTrain.to(model.device if hasattr(model, 'device') else 'cuda')
    y = yTrain.to(model.device if hasattr(model, 'device') else 'cuda')
    
    # Forward pass
    if hasattr(model, 'model'):  # For your Model class wrapper
        yHat, cls = model.model(x)
        loss = model.criterion(yHat, y)
    else:  # Direct model
        yHat, cls = model(x)
        loss = torch.nn.CrossEntropyLoss()(yHat, y)
    
    print(f"Loss value: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"✓ {name}: grad_norm = {grad_norm:.8f}")
                if grad_norm > 1e-10:
                    has_gradients = True
            else:
                print(f"❌ {name}: NO GRADIENT")
    
    if not has_gradients:
        print("❌ PROBLEM: NO MEANINGFUL GRADIENTS!")
    else:
        print("✓ Gradients are flowing")
    
    # 3. Check predictions
    print(f"\n3. Model predictions analysis...")
    probs = torch.softmax(yHat, dim=1)
    preds = yHat.argmax(1)
    
    print(f"Logits: {yHat[0].detach().cpu().numpy()}")
    print(f"Probabilities: {probs[0].detach().cpu().numpy()}")
    print(f"Predictions: {preds.detach().cpu().numpy()}")
    print(f"Ground truth: {y.detach().cpu().numpy()}")
    
    # Check if model is always predicting the same class
    unique_preds = torch.unique(preds)
    print(f"Unique predictions: {unique_preds.cpu().numpy()}")
    
    if len(unique_preds) == 1:
        print("❌ PROBLEM: Model always predicts the same class!")
    
    # 4. Check if loss is at the expected baseline
    n_classes = yHat.shape[1]
    random_loss = -torch.log(torch.tensor(1.0/n_classes))
    print(f"\nExpected random loss: {random_loss.item():.6f}")
    print(f"Actual loss: {loss.item():.6f}")
    
    if abs(loss.item() - random_loss.item()) < 0.01:
        print("❌ PROBLEM: Loss is at random baseline - model might not be learning")

def check_data_loading(dataLoader):
    """
    Check if data is loading correctly
    """
    print("\n=== CHECKING DATA LOADING ===")
    
    batch_count = 0
    label_counts = {}
    
    for i, batch in enumerate(dataLoader):
        if i >= 3:  # Check first 3 batches
            break
            
        batch_count += 1
        xTrain = batch[0]
        yTrain = batch[1]
        
        print(f"Batch {i}:")
        print(f"  Input shape: {xTrain.shape}")
        print(f"  Labels shape: {yTrain.shape}")
        print(f"  Labels: {yTrain.numpy()}")
        
        # Count label distribution
        for label in yTrain.numpy():
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Check for NaN or inf
        if torch.isnan(xTrain).any():
            print("  ❌ NaN values in input!")
        if torch.isinf(xTrain).any():
            print("  ❌ Inf values in input!")
    
    print(f"Total batches checked: {batch_count}")
    print(f"Label distribution: {label_counts}")

def debug_transfer_learning_specific(model):
    """
    Check transfer learning specific issues
    """
    print("\n=== TRANSFER LEARNING SPECIFIC CHECKS ===")
    
    # Check if pretrained weights are actually loaded
    if hasattr(model, 'model'):
        bolt_model = model.model
    else:
        bolt_model = model
    
    # Check classifier head initialization
    classifier = bolt_model.classifierHead
    print(f"Classifier: {classifier}")
    
    if hasattr(classifier, '0'):  # Sequential classifier
        first_layer = classifier[0]
        print(f"First layer weight mean: {first_layer.weight.data.mean().item():.6f}")
        print(f"First layer weight std: {first_layer.weight.data.std().item():.6f}")
        
        # Check if weights look randomly initialized (should be for new head)
        if first_layer.weight.data.std().item() < 0.001:
            print("❌ PROBLEM: Classifier weights might be zero-initialized!")
    
    # Check if frozen layers have reasonable weights (from pretraining)
    sample_frozen_layer = None
    for name, param in bolt_model.named_parameters():
        if not param.requires_grad and 'weight' in name:
            print(f"Frozen layer: {name}")
            sample_frozen_layer = param
            
    
            if sample_frozen_layer is not None:
                frozen_mean = sample_frozen_layer.data.mean().item()
                frozen_std = sample_frozen_layer.data.std().item()
                print(f"Sample frozen layer - mean: {frozen_mean:.6f}, std: {frozen_std:.6f}")
                
                if frozen_std < 0.001:
                    print("❌ PROBLEM: Frozen layers might not be properly loaded!")

def check_optimizer_and_scheduler(model):
    """
    Check optimizer and scheduler setup
    """
    print("\n=== OPTIMIZER & SCHEDULER CHECKS ===")
    
    if hasattr(model, 'optimizer'):
        optimizer = model.optimizer
        print(f"Optimizer: {type(optimizer).__name__}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        print(f"Weight decay: {optimizer.param_groups[0]['weight_decay']}")
        
        # Check if optimizer has parameters
        param_count = sum(len(group['params']) for group in optimizer.param_groups)
        print(f"Parameters in optimizer: {param_count}")
        
        if param_count == 0:
            print("❌ PROBLEM: Optimizer has no parameters!")
    
    if hasattr(model, 'scheduler'):
        scheduler = model.scheduler
        print(f"Scheduler: {type(scheduler).__name__}")

# Usage for your specific case:
def debug_your_model():
    """
    Specific debugging for your transfer learning setup
    """
    print("=== DEBUGGING YOUR TRANSFER LEARNING ===")
    
    # Add this in your train function after model initialization:
    """
    # After loading pretrained model and setting up transfer learning:
    debug_transfer_learning_specific(model)
    check_optimizer_and_scheduler(model)
    
    # After creating dataLoader:
    check_data_loading(dataLoader)
    
    # After first few training steps if loss isn't changing:
    debug_loss_not_changing(model.model, dataLoader)
    """

# Quick fixes to try:

def quick_fixes():
    """
    Quick things to try when loss isn't changing
    """
    print("""
    QUICK FIXES TO TRY:
    
    1. CHECK TRAINABLE PARAMETERS:
       for name, param in model.model.named_parameters():
           if param.requires_grad:
               print(f"Trainable: {name}")
    
    2. SIMPLIFY CLASSIFIER HEAD:
       # Replace complex head with simple linear layer
       model.model.classifierHead = torch.nn.Linear(in_features, num_classes)
    
    3. INCREASE LEARNING RATE TEMPORARILY:
       # Try much higher LR to see if anything happens
       optimizer.param_groups[0]['lr'] = 1e-3
    
    4. CHECK LOSS COMPUTATION:
       # Make sure loss is being computed on the right tensors
       print(f"yHat shape: {yHat.shape}, y shape: {y.shape}")
       print(f"yHat dtype: {yHat.dtype}, y dtype: {y.dtype}")
    
    5. DISABLE SCHEDULER TEMPORARILY:
       # Set scheduler to None to eliminate variable
       model.scheduler = None
    
    6. CHECK INPUT PREPROCESSING:
       # Make sure data preprocessing isn't making all inputs identical
       print(f"Input std: {xTrain.std().item()}")
       print(f"Input mean: {xTrain.mean().item()}")
    
    7. REMOVE FROZEN LAYERS ENTIRELY:
       # Unfreeze everything to see if that helps
       for param in model.model.parameters():
           param.requires_grad = True
    """)

# Add these print statements to your training loop:
def add_debug_prints():
    """
    Add these to your training loop for debugging
    """
    print("""
    # Add these in your train function:
    
    print(f"Step {i}: Loss = {train_loss.item():.6f}")
    print(f"Step {i}: Predictions = {train_preds.numpy()}")
    print(f"Step {i}: Ground truth = {yTrain.numpy()}")
    
    # Check if loss is exactly the same
    if i > 0 and abs(train_loss.item() - prev_loss) < 1e-8:
        print("WARNING: Loss not changing!")
    prev_loss = train_loss.item()
    
    # Check learning rate
    if hasattr(model, 'optimizer'):
        current_lr = model.optimizer.param_groups[0]['lr']
        print(f"Step {i}: Learning rate = {current_lr}")
    """)

# Most likely culprits in your case:
print("""
MOST LIKELY ISSUES FOR YOUR CASE:

1. NO TRAINABLE PARAMETERS
   - Check if reinitialize() is somehow freezing everything
   - Verify optimizer only gets trainable parameters

2. SCHEDULER ISSUES  
   - OneCycleLR might be setting LR to 0
   - Try disabling scheduler temporarily

3. CLASSIFIER HEAD PROBLEMS
   - New head might not be properly initialized
   - Complex head might have vanishing gradients

4. DATA ISSUES
   - All inputs might be identical after preprocessing
   - Labels might be wrong/corrupted

5. OPTIMIZER SETUP
   - Optimizer might not be getting the right parameters
   - Learning rate might be effectively 0

RUN THESE CHECKS IN ORDER:
1. debug_transfer_learning_specific(model)
2. check_optimizer_and_scheduler(model) 
3. debug_loss_not_changing(model.model, dataLoader)
""")