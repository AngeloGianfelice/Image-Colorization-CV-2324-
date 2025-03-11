import config
from tqdm import tqdm
from random import randint
import torch
from utils import plot_loss,plot_images,lab2rgb

class EarlyStopping:
    def __init__(self, patience=10, delta=0, path="checkpoint.pth", verbose=False):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement.
            delta (float): Minimum change in the monitored metric to qualify as an improvement.
            path (str): Path to save the best model checkpoint.
            verbose (bool): Whether to print improvement messages.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = 1.0

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset counter if validation loss improves

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased ({self.best_loss:.6f} → {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)
        self.best_loss = val_loss

def train_model(model,epochs,device,train_loader,val_loader,criterion,optimizer,scheduler):

    tloss_list=[] #for plotting
    vloss_list=[]

    # === Training Loop === #
    for epoch in range(epochs):

        model.train()  # Set model to training mode
        train_loss = 0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)

        for batch_idx,(image_l, image_ab, _) in enumerate(train_progress):

            image_l, image_ab = image_l.to(device), image_ab.to(device)
            optimizer.zero_grad()
            output = model(image_l)
            loss = criterion(output, image_ab)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # Update tqdm description with batch loss
            train_progress.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=False)

        with torch.no_grad():
            for idx, (L, AB, _) in enumerate(val_progress):
                L, AB = L.to(device), AB.to(device)
                output = model(L)
                loss = criterion(output, AB)
                val_loss += loss.item()
                # Update tqdm description with batch loss
                val_progress.set_postfix(loss=loss.item())
    
        val_loss /= len(val_loader)

        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        tloss_list.append(train_loss)
        vloss_list.append(val_loss)

        # Call early stopping
        scheduler(val_loss, model)

        if scheduler.early_stop:
            print("Early stopping triggered. Training stopped.")
            break
    
    print("✅ Training Complete!")
    plot_loss(tloss_list,vloss_list)

def test_model(model, device, test_loader, input_mode):

    model.eval()
    input_imgs=[]
    output_imgs=[]
    gt_imgs=[]

    # Run inference on test images
    dataiter=iter(test_loader)
    L, AB, RGB = next(dataiter)
    L = L.to(device)
    AB = AB.to(device)
    RGB = RGB.to(device)
        
    with torch.no_grad():
        AB_pred = model(L)

    for _ in range(config.NUM_TEST):

        idx=randint(0,min(config.BATCH_SIZE,len(test_loader.dataset)-1))
        output_ab = AB_pred[idx]

        input_l = L[idx] 
        input_rgb = RGB[idx]
        input_ab_sample = AB[idx]

        input_l *= 100 #denormalization


        if input_mode == 'rgb':
    
            input=input_l[0].cpu()
            colorized = lab2rgb(input_l[0].unsqueeze(0),output_ab)
            ground_truth = input_rgb.cpu().permute(1,2,0)

        elif input_mode == 'gray':
            
            input = input_l.cpu().squeeze(0)
            colorized = lab2rgb(input_l,output_ab)
            ground_truth = lab2rgb(input_l,input_ab_sample)

        input_imgs.append(input)
        output_imgs.append(colorized)
        gt_imgs.append(ground_truth)

    plot_images(input_imgs,output_imgs,gt_imgs)
