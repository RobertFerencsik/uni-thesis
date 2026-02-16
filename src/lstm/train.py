import torch
from torch.optim import Adam
from pathlib import Path

class Trainer:
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate = 1e-3,
        max_grad_norm = 1.0,
        save_dir = './data/models'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.criterion = torch.nn.BCELoss()

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.max_grad_norm = max_grad_norm
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for token_ids, attention_mask, labels in self.train_loader:
            token_ids = token_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)
            
            self.optimizer.zero_grad()
            outputs = self.model(token_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        with torch.no_grad():
            for token_ids, attention_mask, labels in self.val_loader:
                token_ids = token_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                outputs = self.model(token_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1

                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy

    def train(
        self,
        num_epochs
    ):
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            val_loss, val_accuracy = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

        self.save_checkpoint(epoch, is_best=False)

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_info': self.model.get_model_info(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pt')
        else:
            torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch+1}.pt')

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint