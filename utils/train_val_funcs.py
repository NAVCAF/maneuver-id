from tqdm.auto import tqdm
import torch


def train(model, train_loader, optimizer, criterion, device):
    """
    DESC
    ---
    Trains a complete epoch for a model and returns the average loss and accuracy
    ---
    INPUTS
    ---
    model: the model being trained
    train_loader: DataLoader instance with training data
    optimizer: optimizer instance used or training
    criterion: loss function instance used for training
    device: device that the model is on
    ---
    RETURN
    ---
    train_loss: average loss over the epoch
    train_acc: average accuracy over the epoch
    """
    model.train()
    # track loss and accuracy over the epoch
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 
    total_loss = 0
    batch_acc = 0
    for i, (x, y, lx) in enumerate(train_loader):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)

        # batch size is first dimension of input
        batch_size = x.shape[0]
        
        # get predictions and calculate loss
        outputs = model(x)
        loss = criterion(outputs,y)
        batch_acc += 100 * int((torch.argmax(outputs, dim=1) == y).sum()) / batch_size
        total_loss += loss

        # backprop
        loss.backward()
        optimizer.step()
        
        # update batch bar
        batch_bar.set_postfix(loss="{:.07f}".format(float(total_loss / (i + 1))),
                                acc="{:.07f}".format(float(batch_acc / (i + 1))))
        batch_bar.update()

    # close batch bar and return loss and accuracy
    batch_bar.close()
    train_loss = float(total_loss / len(train_loader))
    train_acc = float(batch_acc / (len(train_loader)))
    return train_loss, train_acc

def val(model, val_loader, criterion, device):
    """
    DESC
    ---
    Runs validation or test data through the model and 
    returns the average loss and accuracy
    ---
    INPUTS
    ---
    model: the trained model
    val_loader: DataLoader instance with validation or test data
    criterion: loss function instance used for training
    device: device that the model is on
    ---
    RETURN
    ---
    val_loss: average loss over the epoch
    val_acc: average accuracy over the epoch
    """
    model.eval()
    # track loss and accuracy over the epoch
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc='Val/Test') 
    total_loss = 0
    batch_acc = 0
    for i, (x, y, lx) in enumerate(val_loader):
        x = x.to(device)
        y = y.to(device)

        # batch size is first dimension of input
        batch_size = x.shape[0]

        # get predictions and calculate loss without backprop
        with torch.no_grad():
            outputs = model(x)
            loss = criterion(outputs,y)
        batch_acc += 100 * int((torch.argmax(outputs, dim=1) == y).sum()) / batch_size
        total_loss += loss

        # update batch bar
        batch_bar.set_postfix(loss="{:.07f}".format(float(total_loss / (i + 1))),
                                acc="{:.07f}".format(float(batch_acc / (i + 1))))
        batch_bar.update()
    
    # close batch bar and return loss and accuracy
    batch_bar.close()
    val_loss = float(total_loss / len(val_loader))
    val_acc = float(batch_acc / (len(val_loader)))
    return val_loss, val_acc