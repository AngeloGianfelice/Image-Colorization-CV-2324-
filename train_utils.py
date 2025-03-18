import config
from tqdm import tqdm
from random import randint
import torch
from utils import plot_loss,plot_images,lab2rgb,rgb2lab,plot_prediction
import torchvision.transforms as transforms
from PIL import Image
import cv2
import torch.nn.functional as F
import torchmetrics
import numpy as np

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

def train_model(model,epochs,device,train_loader,val_loader,criterion,optimizer,scheduler,fname):

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
    plot_loss(tloss_list,vloss_list,fname=fname)
    with open("loss_log.txt", "w") as f:
        for loss in tloss_list:
            f.write(f"{loss}\n")  # Write each loss value on a new line
        f.write("## ============= ##\n")
        for loss in vloss_list:
            f.write(f"{loss}\n")  # Write each loss value on a new line

def test_model(model, device, test_loader, input_mode):

    model.eval()
    mse_total, psnr_total, ssim_total,delta_e_total = 0.0, 0.0, 0.0, 0.0
    ssim_fn = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    num_images=len(test_loader.dataset)
    graylist,colorlist,gt_list=[],[],[]

    with torch.no_grad():
        for idx,(L, AB, RGB)in enumerate(test_loader):
            L = L.to(device)
            AB = AB.to(device)
            RGB = RGB.to(device)
        
            AB_pred = model(L)

            for i in range(len(L)):

                output_ab = AB_pred[i]
                input_l = L[i] 
                input_rgb = RGB[i]
                input_ab_sample = AB[i]

                input_l *= 100 #denormalization


                if input_mode == 'rgb':
    
                    input=input_l[0].cpu()
                    colorized = lab2rgb(input_l[0].unsqueeze(0),output_ab)
                    ground_truth = np.asarray(input_rgb.cpu().permute(1,2,0))

                elif input_mode == 'gray':
            
                    input = input_l.cpu().squeeze(0)
                    colorized = lab2rgb(input_l,output_ab)
                    ground_truth = lab2rgb(input_l,input_ab_sample)

                if len(graylist)<=config.NUM_TEST:
                    graylist.append(input)
                    colorlist.append(colorized)
                    gt_list.append(ground_truth)
                
                if  len(graylist) == config.NUM_TEST:
                    plot_images(graylist,colorlist,gt_list)
            
                # Compute MSE
                mse = F.mse_loss(torch.tensor(colorized).permute(2,1,0), torch.tensor(ground_truth).permute(2,1,0), reduction="mean").item()
                
                max_pixel_value = 1 #images in [0,1]
                
                psnr = 10 * np.log10(max_pixel_value ** 2 / mse).item()

                # Compute SSIM
                ssim = ssim_fn(torch.tensor(colorized).permute(2,1,0).unsqueeze(0), torch.tensor(ground_truth).permute(2,1,0).unsqueeze(0)).item()

                #compute delta-e
                delta_e = np.mean(np.sqrt(np.sum((cv2.cvtColor(colorized,cv2.COLOR_RGB2LAB) - cv2.cvtColor(ground_truth,cv2.COLOR_RGB2LAB)) ** 2, axis=-1))) 

                # Accumulate results
                mse_total += mse
                psnr_total += psnr 
                ssim_total += ssim  
                delta_e_total += delta_e 

    # Compute mean values
    mse_avg = mse_total / num_images
    psnr_avg = psnr_total / num_images
    ssim_avg = ssim_total / num_images
    delta_e_avg = delta_e_total / num_images

    print(f"MSE: {mse_avg:.4f}, PSNR: {psnr_avg:.4f} SSIM: {ssim_avg:.4f}, ΔE: {delta_e_avg:.4f}")
    print("✅ Testing Complete!")
    return 

def predict(image_path, model, device, input_mode):

    model.eval()

    transform=transforms.Compose([
                transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
                transforms.ToTensor()
             ]) # No augmentation for val/test

    input_image = Image.open(image_path).convert("RGB") 

    augm_image = transform(input_image)
        
    L_channel,AB_channel=rgb2lab(augm_image)
    
    if input_mode == 'rgb':
        l_rgb = cv2.cvtColor(L_channel, cv2.COLOR_GRAY2RGB)  # Shape: (224, 224, 3)
        L_tensor = torch.tensor(l_rgb).permute(2, 0, 1).unsqueeze(0)

    elif input_mode == 'gray':
        L_tensor = torch.tensor(L_channel)[None,None,:,:]
    else: 
        print("Wrong input mode, Exiting...")
        exit()

    AB_tensor = torch.tensor(AB_channel).permute(2, 0, 1)
    L_tensor=L_tensor.to(device)
    AB_tensor=AB_tensor.to(device)
    augm_image=augm_image.to(device)

    with torch.no_grad():
        AB_pred = model(L_tensor)

    L_tensor *= 100 #denormalization

    if input_mode == 'rgb':
    
        input=L_tensor.cpu().squeeze(0)
        colorized = lab2rgb((L_tensor[0][0].unsqueeze(0)),AB_pred[0])
        input=input.permute(1,2,0) / 100

    elif input_mode == 'gray':
        
        input = L_tensor.cpu().squeeze(dim=(0, 1))
        colorized = lab2rgb(L_tensor.squeeze(0),AB_pred.squeeze(0)) 
        
    plot_prediction(input,colorized)

    

        