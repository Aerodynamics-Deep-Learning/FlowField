import torch
import wandb
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error
from auxiliary import get_n_params

def train(cfg_train:(dict), cfg_export:(dict), model:(torch.nn.Module), optimizer:(torch.optim), scheduler, loss_fn, dataloaders:(list)):

    """
    This function is to train the model given an opitimizer, loss, and data.

    Args:
        cfg_train (dict):
            - 'cfg_loader' : The config dict for the loader that was previously used
            - 'device' : The device being used
            - 'dtype': Dtype to be used during the entire training process
        cfg_export (dict):
            - 'model_save': save name for model being trained
            - 'val_export': validation step exports
        model (torch.nn.Model): The DL model being used
        optimizer (torch.optim): The optimizer being used to improve the model
        loss: The loss function
        dataloaders (list): List of dataloaders [dl_train, dl_val, dl_test] 
    
    Returns:
        collector_dict (dict): Dictionary of logs
        model (nn.Module): Trained model
    """

    # Unpack the cfg_train dict
    device = cfg_train['device']
    dtype = cfg_train['dtype']
    cfg_loader = cfg_train['cfg_loader']
    n_epoch = cfg_loader['n_epoch']
    n_train = cfg_loader['n_train']
    train_batch = cfg_loader['train_batch']

    dl_train, dl_val, dl_test = dataloaders[0], dataloaders[1], dataloaders[2] # Unpack the dataloaders
    iter_train = iter(dl_train); iter_val = iter(dl_val); iter_test = iter(dl_test) # Define the iterators for the dataloaders
    
    # Before training, infer the amount of training batches per epoch
    n_train_batch_per_epoch = int(n_train/train_batch)
    
    # Move model to device
    model.to(cfg_train['device'])

    # Setup the save paths
    model_save = cfg_export['model_save']
    if not model_save.endswith('.pth'):
        model_save += '.pth'

    val_save = cfg_export['val_save']
    if not val_save.endswith('.parquet'):
        val_save += '.parquet'
    
    # Log the param amount for model
    wandb.log({
        "model/param#": get_n_params(model=model)
    })

    val_preds=[]
    val_targets=[]
    best_val_loss = float('inf')

    for epoch in range(1, n_epoch +1):

        # Training loop for the epoch
        model.train() # Set the model to training mode
        total_train_loss = 0.0 # Set the training loss to 0

        for _ in range(n_train_batch_per_epoch):

            # Get the batch
            try:
                batch= next(iter_train)
            except StopIteration:
                # If the dataset is exhausted, reset the iterator for the next epoch
                raise StopIteration("The training dataset has been exhausted. There must be a fault.")
            
            # Get the values from the batch, move them to the necessary device with necessary dtype
            shape_input, param_input, target, airfoil_name = batch['geometry'].to(device= device, dtype= dtype), batch['cond'].to(device= device, dtype= dtype), batch['perf_coeffs'].to(device= device, dtype= dtype), batch['name'] 
            # Zero them grads
            optimizer.zero_grad()
            # Get the predictions
            pred = model(shape_input= shape_input, param_input= param_input)
            # Get the loss 
            train_loss = loss_fn(target, pred)
            # Backprop
            train_loss.backward()
            optimizer.step()

            # Log train step results
            total_train_loss += train_loss.item()
            wandb.log({
                "train/loss_batch": train_loss.item(),
                "epoch": epoch
            })

        # Log average train results
        wandb.log({
            "General/train_average_loss": total_train_loss/n_train_batch_per_epoch,
        })

        # Step Scheduler, if exists
        if scheduler is not None:
            scheduler.step()
        
        # Validation step
        # Set model to eval
        model.eval()

        with torch.no_grad():

            try:
                batch= next(iter_val)
            except StopIteration:
                # If the dataset is exhausted, reset the iterator for the next epoch
                raise StopIteration("The validation dataset has been exhausted. There must be a fault.")
        
            # Get data, pred, loss
            shape_input, param_input, target, airfoil_name = batch['geometry'].to(device= device, dtype= dtype), batch['cond'].to(device= device, dtype= dtype), batch['perf_coeffs'].to(device= device, dtype= dtype), batch['name'] 
            pred = model(shape_input= shape_input, param_input= param_input)
            val_loss = loss_fn(target, pred)

            # Get metrics
            r2 = r2_score(target.cpu(), pred.cpu(), multioutput='raw_values')
            mae = mean_absolute_error(target.cpu(), pred.cpu(), multioutput='raw_values')

            # Log the validation loss
            wandb.log({
                "General/val_average_loss": val_loss.item(),
                "val_metrics/r2_cd": r2[0],
                "ValMetrics/r2_cl": r2[1],
                "valMetrics/r2_cm": r2[2],
                "valMetrics/mae_cd": mae[0],
                "valMetrics/mae_cl": mae[1],
                "valMetrics/mae_cm": mae[2],
                "General/epoch": epoch
            })

            # Log the validation predictions and actual vals
            val_preds.append(pred.cpu().numpy())
            val_targets.append(pred.cpu().numpy())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save)
            # Optional: Log best metric to summary
            wandb.run.summary["val/loss_best"] = best_val_loss

    # Since all vals are finished, export preds during val steps
    val_preds_arr = np.concatenate(val_preds, axis=0)
    val_targets_arr = np.concatenate(val_targets, axis=0)
    del val_preds, val_targets

    df_val_pred = pd.DataFrame(val_preds_arr, columns=['Pred_Cd', 'Pred_Cl', 'Pred_Cm'])
    df_val_targets = pd.DataFrame(val_targets_arr, columns=['Target_Cd', 'Target_Cl', 'Target_Cm'])
    df_export = pd.concat([df_val_pred, df_val_targets], axis=1)
    df_export.to_parquet(val_save, engine='pyarrow', compression='snappy', index=False)
    del df_val_pred, df_val_targets, df_export

    # Testing step
    # Load the best model
    model.load_state_dict(torch.load(model_save, map_location='cpu'))

    with torch.no_grad():

        try:
            batch= next(iter_test)
        except StopIteration:
            # If the dataset is exhausted, reset the iterator for the next epoch
            raise StopIteration("The validation dataset has been exhausted. There must be a fault.")
        
        # Get data, pred, loss
        shape_input, param_input, target, airfoil_name = batch['geometry'].to(device= device, dtype= dtype), batch['cond'].to(device= device, dtype= dtype), batch['perf_coeffs'].to(device= device, dtype= dtype), batch['name'] 
        pred = model(shape_input= shape_input, param_input= param_input)
        test_loss = loss_fn(target, pred)

        # Get metrics
        r2 = r2_score(target.cpu(), pred.cpu(), multioutput='raw_values')
        mae = mean_absolute_error(target.cpu(), pred.cpu(), multioutput='raw_values')

        # Log the test loss
        wandb.log({
            "test/loss": test_loss.item(),
            "test/r2_cd": r2[0],
            "test/r2_cl": r2[1],
            "test/r2_cm": r2[2],
            "test/mae_cd": mae[0],
            "test/mae_cl": mae[1],
            "test/mae_cm": mae[2],
        })

        # Log the validation predictions and actual vals
        df_test_pred = pd.DataFrame(pred.cpu().numpy(), columns=['Pred_Cd', 'Pred_Cl', 'Pred_Cm'])
        df_test_target = pd.DataFrame(target.cpu().numpy(), columns=['Target_Cd', 'Target_Cl', 'Target_Cm'])
        df_test = pd.concat([df_test_pred, df_test_target], axis=1)
        test_table = wandb.Table(dataframe=df_test)
        wandb.log({
            "test/results_table": test_table
        })
        del test_table, df_test, df_test_pred, df_test_target

    """
    Train, validation, and test has been finished.
    """

    return model