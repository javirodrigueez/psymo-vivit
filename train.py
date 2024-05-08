"""
Train ViViT model on Psymo dataset.

Usage:
    train.py [options] [-h]

Options:
    --checkpoint_path=<checkpoint_path>     Path to a checkpoint to load the model from
    --data_path=<data_path>     Dataset to use [default: /data/psymo-data/]
    --annot_path=<annot_path>   Annotations path [default: /data/psymo-annot/]
    --trait_name=<trait_name>       Trait name [default: BFI_Openness]
    --batch_size=<batch_size>   Batch size [default: 8]
    --epochs=<epochs>           Number of epochs [default: 25]
    --freeze                    Freeze the feature extractor
    --eval                     Only evaluate the model
    -h --help                   Show this screen
"""

import torch, wandb
from torch.utils.data import DataLoader
from transformers import VivitForVideoClassification, VivitImageProcessor, VivitConfig
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from dataset import GaitDataset
from docopt import docopt
from tqdm import tqdm
from statistics import mean
import time
from sklearn.metrics import f1_score, accuracy_score, precision_score
import os

def collate_fn(batch):
    """
    batch: una lista de tuplas (secuencia, etiqueta)
    donde 'secuencia' es un tensor de PyTorch de frames y 'etiqueta' es la etiqueta de la secuencia.
    """
    # Desempaquetar los datos del batch
    sequences, labels = zip(*batch)
    max_frames = 16

    # Inicializa una lista para almacenar las secuencias procesadas
    processed_sequences = []

    for s in sequences:
        if s.size(0) > max_frames:
            # Si la secuencia tiene más de max_frames frames, selecciona max_frames frames de forma uniforme
            indices = torch.linspace(0, s.size(0) - 1, max_frames).long()
            subsampled_sequence = s[indices]
            processed_sequences.append(subsampled_sequence)
        else:
            # Si la secuencia tiene menos de max_frames frames, agregar padding
            padding = torch.zeros(max_frames - s.size(0), *s.shape[1:])
            padded_sequence = torch.cat([s, padding], dim=0)
            processed_sequences.append(padded_sequence)

    # Stack las secuencias procesadas en un nuevo tensor
    padded_sequences = torch.stack(processed_sequences)

    # Stack las etiquetas también, si las etiquetas son tensores; si son números o categorías, usa torch.tensor(labels)
    labels = torch.tensor(labels)

    return padded_sequences, labels

def main(args):
    # Init
    data_path = args['--data_path']
    annot_path = args['--annot_path']
    trait_name = args['--trait_name']
    freeze = args['--freeze']
    eval = args['--eval']
    checkpoint_path = args['--checkpoint_path']
    batch_size = int(args['--batch_size'])
    print(f"Training model for trait: {trait_name}")
    print('Freeze feature extractor:', freeze)
    # Load data
    train_dataset = GaitDataset(data_path=data_path, annot_path=annot_path, trait_name=trait_name, samples_size=25) #125
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    val_dataset = GaitDataset(data_path=data_path, annot_path=annot_path, trait_name=trait_name, split='val', samples_size=7) # 30
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    # Load and customize model
    config = VivitConfig(num_frames=16, num_hidden_layers=6, num_attention_heads=6)
    model = VivitForVideoClassification(config)

    if 'BFI' in trait_name or 'BPAQ' in trait_name or 'OFER' in trait_name:
        output_classes = 4
    elif 'RSE' in trait_name or 'GHQ' in trait_name:
        output_classes = 3
    elif 'DASS' in trait_name:
        output_classes = 5

    model.classifier = torch.nn.Linear(model.config.hidden_size, output_classes)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))

    if freeze:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    # Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = int(args['--epochs'])
    max_val_accuracy = 0.0

    model.train()
    for epoch in range(epochs):
        start = time.time()
        if not eval:
            epoch_losses = []
            for inputs, labels in tqdm(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs['pixel_values'] = inputs['pixel_values'].squeeze()

                outputs = model(**inputs)
                loss = criterion(outputs.logits.float(), labels)
                epoch_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            #total_correct = 0
            #total = 0
            val_loss = []
            all_predictions = []
            all_labels = []

            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs['pixel_values'] = inputs['pixel_values'].squeeze()
                outputs = model(**inputs)
                loss = criterion(outputs.logits, labels)
                val_loss.append(loss.item())

                _, predicted = torch.max(outputs.logits, 1)

                # Collect all predictions and labels for computing accuracy, precision, f1-score
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Compute metrics
            accuracy = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, average='macro')
            f1 = f1_score(all_labels, all_predictions, average='macro')

            # Save best model if accuracy is higher
            if accuracy > max_val_accuracy:
                max_val_accuracy = accuracy

                if not os.path.exists(f'weights/{trait_name}'):
                    os.makedirs(f'weights/{trait_name}')
                torch.save(model.state_dict(), f'weights/{trait_name}/best_model_epoch_{epoch}.pth')

        # Show metrics
        end = time.time()
        print(f"Epoch {epoch}")
        print(f"Validation Accuracy: {accuracy}")
        print(f"Validation Precision: {precision}")
        print(f"Validation F1-score: {f1}")
        print(f"Validation Loss: {mean(val_loss)}")
        if not eval:
            print(f"Training loss: {mean(epoch_losses)}")
        print(f"Time: {end-start}")
        if not eval:
            wandb.log({
                'Validation Accuracy': accuracy,
                'Validation Precision': precision,
                'Validation F1-score': f1, 
                'Validation Loss': mean(val_loss), 
                'Training loss': mean(epoch_losses), 
                'Epoch time': end-start
            })
    
    # Finish training
    print('Training finished')
    torch.save(model.state_dict(), 'weights/final_model.pth')
    wandb.finish()



if __name__ == '__main__':
    args = docopt(__doc__)
    #wandb.init(project='psymo-vivit', name=args['--trait_name'])
    wandb.init(mode='disabled')
    main(args)