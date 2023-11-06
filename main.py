import torch
from generator.trainer import Trainer
from generator.generator import TextGenerator
from classifier.classifier import BertClassifier
from generator.dataPreprocessor import DataPreprocessor
import os

def main():

    data_path = 'classifier/model_datasets/IMDB Dataset.csv'
    data_pre = DataPreprocessor(data_path)
    sos_index = data_pre.vocab['<sos>']
    eos_index = data_pre.vocab['<eos>']
    train_data, test_data = data_pre.split_data()
    train_dataset, test_dataset = data_pre.create_datasets(train_data, test_data)
    # Hyperparameters and configurations
    vocab_size = len(data_pre.get_vocab())
    embed_size = 256
    control_embed_size = 256
    num_layers = 6
    heads = 8
    batch_size = 32
    epochs = 10
    learning_rate = 0.001

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the components
    text_generator = TextGenerator(vocab_size, embed_size, control_embed_size, num_layers, heads, sos_index, eos_index, device).to(device)
    classifier_model = BertClassifier().to(device)
    classifier_model.load_state_dict(torch.load('classifier/classifier_3labels.pth'))

    # Assuming train_dataset and val_dataset are instances of torch.utils.data.Dataset
    trainer = Trainer(text_generator, classifier_model, train_dataset, test_dataset, learning_rate)

    # Train the model
    trainer.train(epochs, batch_size)

    # Save the model
    model_path = 'text_generator_model.pth'
    trainer.save_model(model_path)


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    main()
