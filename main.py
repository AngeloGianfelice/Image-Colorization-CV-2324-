import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import config
from colorizers import ColorizationAutoencoder,ResnetAutoencoder
from datasets import Cocostuff_Dataset
from train_utils import EarlyStopping,train_model,test_model,predict
import argparse

def main():

    # Print used device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print all variables defined in config.py
    print("\n# === Chosen Hyperparameters === #")
    print(f"DEVICE = {torch.cuda.get_device_name()}")
    for attr in dir(config):
        if not attr.startswith("__"):  # Ignore built-in attributes
            print(f"{attr} = {getattr(config, attr)}")
    print("# ============================== #\n")

    parser = argparse.ArgumentParser(description="Parse two input arguments")
    
    # Add arguments
    parser.add_argument("--model", type=str, required=True, default="model1", help="model1 or model2")
    parser.add_argument("--mode", type=str, required=True, default="train", help="train or test")
    parser.add_argument("--image_path", type=str, help="path of image to colorize")

    # Parse arguments
    args = parser.parse_args()

    if args.model == 'model1':
        model = ColorizationAutoencoder().to(device)
        input_mode='gray'

    elif args.model == 'model2':
        model = ResnetAutoencoder().to(device)
        input_mode='rgb'

    else:
        print("wrong model chosen, Exiting...")
        return
    
    model_path=f"models/{args.model}_{config.EPOCHS}.pth"
    loss_fname=f"plots/{args.model}_{config.EPOCHS}_loss.png"

    if args.mode == 'train':

        # Create datasets
        train_dataset = Cocostuff_Dataset(config.DATASET_PATH, phase="train",input_mode=input_mode)
        val_dataset = Cocostuff_Dataset(config.DATASET_PATH, phase="val",input_mode=input_mode)
        test_dataset = Cocostuff_Dataset(config.DATASET_PATH, phase="test",input_mode=input_mode)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,num_workers=4,pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,num_workers=4,pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,num_workers=4,pin_memory=True)

        print(f"✅ Dataset Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

        # === Define Training Components === #
        criterion = nn.MSELoss()  # Loss Function
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)  # Optimizer
        early_stopping = EarlyStopping(patience=config.PATIENCE, path=model_path, verbose=True) # Early Stopping

        # === Train Colorization Model === #
        train_model(model=model, epochs=config.EPOCHS, device=device, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, scheduler=early_stopping,fname=loss_fname)

        # === Testing Model === #
        test_model(model=model, device=device, test_loader=test_loader,input_mode=input_mode)

    elif args.mode == 'test':

        test_dataset = Cocostuff_Dataset(config.DATASET_PATH, phase="test",input_mode=input_mode)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,num_workers=4,pin_memory=True)
        print(f"✅ Test images={len(test_dataset)}")

        # Load Best Model for Testing
        model.load_state_dict(torch.load(model_path))
        test_model(model=model, device=device, test_loader=test_loader, input_mode=input_mode)

    elif args.mode == 'predict':
        # Load Best Model for Testing
        model.load_state_dict(torch.load(model_path))
        predict(image_path=args.image_path, model=model, device=device, input_mode=input_mode)
        
    else:
        print("wrong mode chosen, Exiting...")
        return
        
if __name__ == '__main__':
    main()
