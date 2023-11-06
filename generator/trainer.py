import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, classifier, train_dataset, test_dataset, learning_rate):
        self.model = model
        self.classifier = classifier
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.classification_criterion = nn.CrossEntropyLoss()
        self.control_code = 1

    def train(self, epochs, batch_size):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            total_classification_loss = 0
            for (src, src_mask, tgt) in self.train_dataset:
                src_mask = src_mask.to(self.model.device)  # Generate src_mask
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(self.model.device)  # Generate tgt_mask
                src = src.to(self.model.device)
                tgt = tgt.to(self.model.device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                preds = self.model(src, self.control_code, tgt_input, src_mask, tgt_mask, batch_size)

                # compute loss of generated text
                loss = self.criterion(preds.view(-1, preds.size(-1)), tgt_output.view(-1))

                # compute loss of classification
                generated_text = self.generate_text(src, src_mask, max_len=preds.size(1))
                classification_preds = self.classifier(generated_text)
                neutral_label = torch.ones_like(classification_preds).long()
                classification_loss = self.classification_criterion(classification_preds, neutral_label)

                # update parameters
                self.optimizer.zero_grad()
                total_loss = loss + classification_loss  # combine loss
                total_loss.backward()
                self.optimizer.step()
                self.optimizer.step()

                total_loss += loss.item()
                total_classification_loss += classification_loss.item()

            avg_loss = total_loss / len(self.train_dataset)
            avg_classification_loss = total_classification_loss / len(self.train_dataset)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss}, Classification Loss: {avg_classification_loss}")
            self.evaluate(batch_size)

    def evaluate(self, batch_size):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for (src, src_mask, tgt) in self.test_dataset:
                src_mask = src_mask.to(self.model.device)  # Generate src_mask
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(self.model.device)  # Generate tgt_mask
                src = src.to(self.model.device)
                tgt = tgt.to(self.model.device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                preds = self.model(src, self.control_code, tgt_input, src_mask, tgt_mask, batch_size)
                loss = self.criterion(preds.view(-1, preds.size(-1)), tgt_output.view(-1))
                total_loss += loss.item()
        avg_loss = total_loss / len(self.test_dataset)
        print(f"Validation Loss: {avg_loss}")

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def generate_text(self, src, src_mask, max_len):
        self.model.eval()
        generated_text = self.model.greedy_decode(src, src_mask, max_len)
        return generated_text
