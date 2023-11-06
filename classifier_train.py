import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from dataset import ClassifierDataset


class BertTrainer:
    def __init__(self, model, learning_rate):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion.to(self.device)

    def train(self, train_data, test_data, epochs, batch_size):
        train_dataset = ClassifierDataset(train_data)
        test_dataset = ClassifierDataset(test_data)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

        for epoch_num in range(epochs):
            total_acc_train, total_loss_train = self._train_epoch(train_dataloader)
            total_acc_val, total_loss_val = self._test(test_dataloader)

            print(f'''Epoch: {epoch_num + 1} 
                  | Train Loss: {total_loss_train / len(train_data):.3f} 
                  | Train Accuracy: {total_acc_train / len(train_data):.3f} 
                  | Val Loss: {total_loss_val / len(test_data):.3f} 
                  | Val Accuracy: {total_acc_val / len(test_data):.3f}''')

    def _train_epoch(self, dataloader):
        self.model.train()
        total_acc, total_loss = 0, 0
        for batch_input, batch_label in tqdm(dataloader):
            batch_label = batch_label.to(self.device).long()
            mask = batch_input['attention_mask'].to(self.device)
            input_id = batch_input['input_ids'].squeeze(1).to(self.device)

            output = self.model(input_id, mask)
            batch_loss = self.criterion(output, batch_label)
            total_loss += batch_loss.item()

            acc = (output.argmax(dim=1) == batch_label).sum().item()
            total_acc += acc

            self.model.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

        return total_acc, total_loss

    def _test(self, dataloader):
        self.model.eval()
        total_acc, total_loss = 0, 0
        with torch.no_grad():
            for batch_input, batch_label in dataloader:
                batch_label = batch_label.to(self.device).long()
                mask = batch_input['attention_mask'].to(self.device)
                input_id = batch_input['input_ids'].squeeze(1).to(self.device)

                output = self.model(input_id, mask)
                batch_loss = self.criterion(output, batch_label)
                total_loss += batch_loss.item()

                acc = (output.argmax(dim=1) == batch_label).sum().item()
                total_acc += acc

        return total_acc, total_loss

