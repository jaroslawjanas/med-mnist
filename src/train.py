import torch
from tqdm import tqdm

###
# ALL CODE IN THIS FILE HAS BEEN DEPRECATED IN FAVOR OF THE LIGHTNING FRAMEWORK
# https://lightning.ai/docs/pytorch/stable/
##

# Function to calculate accuracy and average loss
def calculate_stats(model, dataloader, criterion, device):
    
    total_loss = 0.0
    correct = 0
    total_samples = 0

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move to device

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class

            # Loss
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)  # Accumulate batch loss

            # Accuracy
            correct += (predicted == labels).sum().item()  # Correct predictions
            total_samples += labels.size(0)  # Total number of samples

    # Compute average loss and accuracy
    accuracy = 100 * correct / total_samples
    loss = total_loss / total_samples

    return accuracy, loss


class EarlyStopping:
    def __init__(self, stat_to_track, mode="min", patience=5, delta=0.0, delay=0, save_path="best_model.pth"):
        """
        Initialize early stopping parameters.

        Args:
        - stat_to_track: (str) The stat to track, e.g., "val_loss", "val_acc".
        - mode: (str) "min" to minimize the stat, "max" to maximize it.
        - patience: (int) How many epochs to wait before stopping.
        - delta: (float) Minimum change to register as an improvement.
        - delay: (int) Number of epochs to wait before activating early stopping.
        - save_path: (str) Filepath to save the best model.
        """
        assert mode in ["min", "max"], "Mode must be 'min' or 'max'."
        self.stat_to_track = stat_to_track
        self.mode = mode
        self.patience = patience
        self.delta = delta
        self.delay = delay
        self.save_path = save_path

        self.best_stat = float("inf") if mode == "min" else -float("inf")
        self.best_epoch = 0
        self.wait = 0  # Number of epochs with no improvement
        self.stopped_epoch = None  # Epoch when stopping occurred
        self.evidence = None  # Evidence for stopping
        self.best_model_state = None  # Best model state

    def __call__(self, model, stats):
        """
        Check if training should stop based on the tracked stat.

        Args:
        - model: (torch.nn.Module) The model being trained.
        - stats: (dict) Dictionary of metrics from the current epoch.

        Returns:
        - (bool) True if training should stop, False otherwise.
        """
        # Check if delay period has passed
        current_epoch = stats["epoch"]
        if current_epoch < self.delay:
            return False  # Don't activate early stopping during the delay period

        current_stat = stats[self.stat_to_track]
        if self._is_improvement(current_stat):
            self.best_stat = current_stat
            self.best_epoch = stats["epoch"]
            self.wait = 0

            # Save the best model state
            self.best_model_state = model.state_dict()
            torch.save(self.best_model_state, self.save_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = stats["epoch"]
                self.evidence = {
                    "reason": f"No significant improvement in '{self.stat_to_track}'",
                    "best_value": self.best_stat,
                    "best_epoch": self.best_epoch + 1,  # Report 1-based epoch
                    "current_epoch": stats["epoch"] + 1,
                    "patience": self.patience,
                    "delay": self.delay,
                    "saved_model_path": self.save_path,
                }
                return True
        return False

    def _is_improvement(self, current_stat):
        """
        Check if the current stat is an improvement.

        Returns:
        - (bool) True if improved by at least `delta`, False otherwise.
        """
        if self.mode == "min":
            return (self.best_stat - current_stat) > self.delta
        else:  # mode == "max"
            return (current_stat - self.best_stat) > self.delta

    def report(self):
        """
        Provide the reason and evidence for stopping.

        Returns:
        - (dict) A dictionary containing stopping details, or None if training hasn't stopped.
        """
        if self.stopped_epoch is not None:
            return self.evidence
        return {"status": "Training is ongoing or has completed without early stopping."}


def train_loop(model, criterion, optimizer, scheduler, epochs, train_loader, validation_loader, device, stop_fn=None):
    
    model.to(device)
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        total_batches = len(train_loader)  # Get the total number of batches

        stats = {
            "epoch": epoch,
            "loss": 0.0,
            "acc": 0.0,
            "val_loss": 0.0,
            "val_acc": 0.0
        }

        # Training loop with progress bar
        with tqdm(total=total_batches, desc=f"{epoch+1:>3}/{epochs}", unit=" batch") as pbar:
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                # Calculate average loss up to this batch
                stats["loss"] = running_loss / (i + 1) # average loss
                 
                # Calculate predictions and update running corrects (also up to this batch)
                _, preds = torch.max(outputs, 1)  # Get predicted class indices
                running_corrects += (preds == labels).sum().item()  # Count correct predictions
                total_samples += labels.size(0)  # Update total samples
                stats["acc"] = 100 * running_corrects / total_samples # average accuracy

                # Displays this batch's stats
                pbar.set_postfix({
                    "loss": f"{stats['loss']:.3f}",
                    "acc": f"{stats['acc']:.2f}%"
                })

                if i == total_batches - 1:
                    stats["val_acc"], stats["val_loss"] = calculate_stats(model, validation_loader, criterion, device)

                    pbar.set_postfix({
                        "loss": f"{stats['loss']:.3f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                        "acc": f"{stats['acc']:.2f}%",
                        "val_loss": f"{stats['val_loss']:.3f}",
                        "val_acc": f"{stats['val_acc']:.2f}%"
                    })         
                
                pbar.update(1)

        if stop_fn and stop_fn(model, stats):  # Call the stop_fn
            print(f"Stopping early at epoch {epoch+1} due to stop condition.")
            break

        # Step the scheduler at the end of each epoch
        scheduler.step()