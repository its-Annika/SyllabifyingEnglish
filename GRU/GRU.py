import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from itertools import product

#vocabulary class
class Vocabulary:
    #don't need an UNK, as the entire alphabet is in training
    def __init__(self, pad_token="[PAD]"):
        #how to get from grapheme to the idx
        self.graph2idx = {pad_token: 0}
        #how to get from the idx to the grapheme
        self.idx2graph = {0: pad_token}
        #what the PAD and its index
        self.pad_token = pad_token
        self.pad_idx = 0
    
    #add a token to the vocabulary
    def add_token(self, token):
        if token not in self.graph2idx:
            self.graph2idx[token] = len(self.graph2idx)
            self.idx2graph[len(self.idx2graph)] = token
    
    #get the length of the vocabulary
    def __len__(self):
        return len(self.graph2idx)
    
    #get the id given the grapheme
    def token_to_idx(self, token):
        #no default -- no UNKs
        return self.graph2idx.get(token)
    
    #get the grapheme given the id
    def idx_to_token(self, idx):
        return self.idx2graph.get(idx)
    
    #turns a list of graphemes into a list of indices
    def tokens_to_indices(self, tokens):
        return [self.token_to_idx(token) for token in tokens]
    
    #turns a list of indicies into a list of graphemes
    def indices_to_tokens(self, indices):
        return [self.idx_to_token(idx) for idx in indices]

def build_vocab(sentences, min_freq=2): 
    word_vocab = Vocabulary()
    tag_vocab = Vocabulary(pad_token="[PAD]")
    
    word_counter = Counter()
    
    #see how often each word appears
    for tokens, tags in sentences:
        for token in tokens:
            word_counter[token.lower()] += 1
        for tag in tags:
            tag_vocab.add_token(tag)
    
    #only add words to the vocabulary that appear at least min_freq
    for word, count in word_counter.items():
        if count >= min_freq:
            word_vocab.add_token(word)
    
    return word_vocab, tag_vocab

#data set class
class graphDataset(Dataset):
    def __init__(self, graphemeForm, graph_vocab, tag_vocab):
        self.graphemeForms = graphemeForm
        self.graph_vocab = graph_vocab
        self.tag_vocab = tag_vocab
    
    #get the length of the dataset
    def __len__(self):
        return len(self.graphemeForms)
    
    #get the sentences at a specific ID
    def __getitem__(self, idx):
        tokens, tags = self.graphemeForms[idx]
        token_indices = self.graph_vocab.tokens_to_indices(tokens)
        tag_indices = self.tag_vocab.tokens_to_indices(tags)
        
        return {
            'tokens': torch.tensor(token_indices),
            'tags': torch.tensor(tag_indices),
            'lengths': len(tokens)
        }

#function that does padding 
def collate_fn(batch):
    #find the largest item in the batch
    max_len = max([item['lengths'] for item in batch])
    
    batch_tokens = []
    batch_tags = []
    batch_lengths = []
    
    for item in batch:
        tokens = item['tokens']
        tags = item['tags']
        length = item['lengths']
        
        #add 0s to the end of the list of tokens and tags (if smaller than the max)
        padded_tokens = torch.cat([tokens, torch.zeros(max_len - len(tokens), dtype=torch.long)])
        padded_tags = torch.cat([tags, torch.zeros(max_len - len(tags), dtype=torch.long)])
        
        #put everything back
        batch_tokens.append(padded_tokens)
        batch_tags.append(padded_tags)
        batch_lengths.append(length)
    
    return {
        'tokens': torch.stack(batch_tokens),
        'tags': torch.stack(batch_tags),
        'lengths': torch.tensor(batch_lengths)
    }

#the rnn class
class SyllableRNN(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(SyllableRNN, self).__init__()

        #size of the hidden layer
        self.hidden_dim = hidden_dim
        
        #chracter embedding layer 
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        #3 layer bi-directional GRU with dropout
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=3, batch_first=True, dropout=0.2, bidirectional=True)

        #dropout applied after last gru layer
        self.dropout = nn.Dropout(0.5)
       
        #layer normalization 
        self.norm = nn.LayerNorm(2 * hidden_dim)

        #time-distributed, fully-connected layer with ReLUC activation 
        self.time_distributed_fc = nn.Linear(2 * hidden_dim, tagset_size)

        #relu activation
        self.ReLU = nn.ReLU()

        #a linear layer
        self.linear = nn.Linear(tagset_size, tagset_size)
    
    def forward(self, graph_ids):

        batch_size, max_seq_length = graph_ids.shape

        #apply the embedding layer to get word vectors
        embeddings = self.embedding(graph_ids)

        #padding stuff
        mask = (graph_ids !=0).float()
        seq_lengths = mask.sum(dim=1).long()

        packed_input = nn.utils.rnn.pack_padded_sequence(embeddings, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        #50% drop out
        output = self.dropout(output)

        #layer norm
        output = self.norm(output)

        #time-distributed, fully connected
        output_reshaped = output.contiguous().view(batch_size * max_seq_length, -1)

        # torch doesn't have a tdfc layer
        #so reshape the input so the fc layer can process the input in time steps
        #with ReLU activation
        output_dense = self.ReLU(self.time_distributed_fc(output_reshaped))

        # then put it back 
        output_dense = output_dense.view(batch_size, max_seq_length, -1)

        #finish with a linear layer
        outputs = self.linear(output_dense)
            
        return outputs

#trains the model
def train(model, train_loader, val_loader, args, device):
    # Very important! Providing ignore_index=0 will ignore the padding token
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    #best_val_acc = 0
    
    #for each epoch
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        #for each batch
        for batch in train_loader:
            tokens = batch['tokens'].to(device)
            tags = batch['tags'].to(device)
            tag_scores = model(tokens)
            
        
            # Turn [batch_size, seq_len, tag_size] into [batch_size * seq_len, tag_size]
            unrolled_tag_scores = tag_scores.view(-1, tag_scores.shape[-1])
            # Turn [batch_size, seq_len] into [batch_size * seq_len]
            unrolled_tags = tags.view(-1)
            loss = criterion(unrolled_tag_scores, unrolled_tags)
            
            #back propigation
            loss.backward()
            #adjust the weights
            optimizer.step()
            #zero out the gradient
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        val_acc, _ = evaluate(model, val_loader, device)
        
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     torch.save(model.state_dict(), args.model_path)
        #     print(f'Model saved to {args.model_path}')
    
    return val_acc

#test the model
def evaluate(model, data_loader, device):
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            tokens = batch['tokens'].to(device)
            tags = batch['tags'].to(device)
            lengths = batch['lengths']
            
            tag_scores = model(tokens)
            
            _, preds = tag_scores.max(dim=2)
            
            for i, length in enumerate(lengths):
                all_preds.extend(preds[i, :length].tolist())
                all_targets.extend(tags[i, :length].tolist())
    
    #see how many predictions match the gold
    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    total = len(all_preds)
    
    return correct / total if total > 0 else 0, (all_preds, all_targets)

#find the numbers for the confusion matrix
def generate_confusion_matrix(model, data_loader, tag_vocab, device):
    model.eval()
    
    num_tags = len(tag_vocab)
    conf_matrix = np.zeros((num_tags, num_tags), dtype=np.int64)
    
    with torch.no_grad():
        for batch in data_loader:
            tokens = batch['tokens'].to(device)
            tags = batch['tags'].to(device)
            lengths = batch['lengths']
            
            tag_scores = model(tokens)
            
            _, preds = tag_scores.max(dim=2)
            
            for i, length in enumerate(lengths):
                for j in range(length):
                    true_tag = tags[i, j].item()
                    pred_tag = preds[i, j].item()
                    if true_tag != 0:
                        conf_matrix[true_tag, pred_tag] += 1
    
    #remove the pad tag 
    conf_matrix = conf_matrix[1:, 1:]
    return conf_matrix

#plot the confusion matrix
def plot_confusion_matrix(conf_matrix, tag_vocab, normalize=True):
    if normalize:
        conf_matrix = conf_matrix.astype('float') / (conf_matrix.sum(axis=1, keepdims=True) + 1e-6)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    #tags = [tag_vocab.idx_to_token(i) for i in range(len(tag_vocab)-1) if tag_vocab.idx_to_token(i) != "[PAD]"]
    tags = [0,1]
    tick_marks = np.arange(len(tags))
    plt.xticks(tick_marks, tags, rotation=90)
    plt.yticks(tick_marks, tags)
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.savefig('GRU/confusion_matrix.png', bbox_inches='tight')
    plt.close()

#print the model errors
def write_all_errors(model, data_loader, graph_vocab, tag_vocab, device):

    model.eval()
    
    # ANSI color codes
    RESET = "\033[0m"
    RED = "\033[31m"
    
    # Function to check if color is supported
    def supports_color():
        import os, sys
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and 'TERM' in os.environ
    
    has_color = supports_color()
    
    with open("GRU/allErrors.txt", 'w') as f:
        for batch in data_loader:
            tokens = batch['tokens']
            true_tags = batch['tags']
            lengths = batch['lengths']
            
            # Get model predictions for this batch
            with torch.no_grad():
                tag_scores = model(tokens.to(device))
                _, pred_tags = tag_scores.max(dim=2)
                pred_tags = pred_tags.cpu()
            
            # Go through each sentence in the batch
            for i, length in enumerate(lengths):
                sentence_has_error = False
                
                # Check if this sentence has any errors
                for j in range(length):
                    if true_tags[i, j] != pred_tags[i, j] and true_tags[i, j] != 0:
                        sentence_has_error = True
                        break
                
                if sentence_has_error:
                    # Get the token strings from indices
                    token_strs = [graph_vocab.idx_to_token(idx.item()) for idx in tokens[i, :length]]
                    true_tag_strs = [tag_vocab.idx_to_token(idx.item()) for idx in true_tags[i, :length]]
                    pred_tag_strs = [tag_vocab.idx_to_token(idx.item()) for idx in pred_tags[i, :length]]
                    
                    f.write("\nword:  " + "".join(token_strs))
                    f.write("\nGold:  ")
                    for t, tag in zip(token_strs, true_tag_strs):
                        f.write(t + "/" + tag + " ")
                    
                    f.write("\nPred:  ")
                    for t, true_tag, pred_tag in zip(token_strs, true_tag_strs, pred_tag_strs):
                        if true_tag != pred_tag:
                                f.write(t.upper() + "/" + pred_tag + " ")
                        else:
                            f.write(t + "/" + pred_tag + " ")
                    f.write("\n")
    f.close()

#print all the model output
def write_all_output(model, data_loader, graph_vocab, tag_vocab, device):
    
    model.eval()
    with open("GRU/allOutput.txt", 'w') as f:
        f.write("notSyllabified\tnumeric\n")
        for batch in data_loader:
            tokens = batch['tokens']
            true_tags = batch['tags']
            lengths = batch['lengths']
            
            # Get model predictions for this batch
            with torch.no_grad():
                tag_scores = model(tokens.to(device))
                _, pred_tags = tag_scores.max(dim=2)
                pred_tags = pred_tags.cpu()
            
            for i, length in enumerate(lengths):
                # Get token and tag strings
                token_strs = [graph_vocab.idx_to_token(idx.item()) for idx in tokens[i, :length]]
                pred_tag_strs = [tag_vocab.idx_to_token(idx.item()) for idx in pred_tags[i, :length]]

                    
                f.write("".join(token_strs) + '\t ' + "".join(pred_tag_strs) + "\n")
    f.close()
                
def parse_args():
    parser = argparse.ArgumentParser(description='RNN POS Tagger')
    
    parser.add_argument('--data_dir', type=str, default='processedData', help='Directory with CoNLL-U files')
    parser.add_argument('--model_path', type=str, default='GRU/GRU_Syllable_RNN.pt', help='Path to save model')
    parser.add_argument('--embedding_dim', type=int, default=100, help='Word embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='LSTM hidden dimension')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--min_freq', type=int, default=2, help='Minimum word frequency')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

 
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("MPS is available -- Using Apple GPU.")
    else:
        device = torch.device("cpu")
        print("MPS not available -- Using CPU.")
    
    train_file = os.path.join(args.data_dir, 'train.tsv')
    dev_file = os.path.join(args.data_dir, 'dev.tsv')
    test_file = os.path.join(args.data_dir, 'test.tsv')

    train_pairs = pd.read_csv(train_file, header=0, sep="\t", usecols=['notSyllabified', 'numeric'], dtype=str)
    train_pairs = list(train_pairs.itertuples(index=False, name=None))
    dev_pairs = pd.read_csv(dev_file, header=0, sep="\t", usecols=['notSyllabified', 'numeric'], dtype=str)
    dev_pairs = list(dev_pairs.itertuples(index=False, name=None))
    test_pairs = pd.read_csv(test_file, header=0, sep="\t", usecols=['notSyllabified', 'numeric'], dtype=str)
    test_pairs = list(test_pairs.itertuples(index=False, name=None))
    
    print(f'Train: {len(train_pairs)} sentences')
    print(f'Dev: {len(dev_pairs)} sentences')
    print(f'Test: {len(test_pairs)} sentences')
    
    graph_vocab, tag_vocab = build_vocab(train_pairs, min_freq=args.min_freq)
    print(f'Vocab size: {len(graph_vocab)} words, {len(tag_vocab)} tags')
    
    train_dataset = graphDataset(train_pairs, graph_vocab, tag_vocab)
    dev_dataset = graphDataset(dev_pairs, graph_vocab, tag_vocab)
    test_dataset = graphDataset(test_pairs, graph_vocab, tag_vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    #adding a manual gridSearch loop:
    # embedding_dims = [32, 64, 96]
    # hidden_dims = [96, 128, 192]
    # l_rs = [0.01, 0.001, 0.0001]
    # epochs = [5,10,15]

    embedding_dims = [32]
    hidden_dims = [96]
    l_rs = [0.01]
    epochs = [1]

    best_acc = 0
    best_config = None
    best_model = None

    index = 0

    for emb_dim, hid_dim, lr, epochs in product(embedding_dims, hidden_dims, l_rs, epochs):
        print(f"Round: {index}")
        print(f"Trying: emb_dim={emb_dim}, hid_dim={hid_dim}, lr={lr}, epochs={epochs}")

        model = SyllableRNN(len(graph_vocab), len(tag_vocab), emb_dim, hid_dim).to(device)
    
        args.embedding_dim = emb_dim
        args.hidden_dim = hid_dim
        args.lr = lr
        args.epochs = epochs
        
        acc = train(model, train_loader, dev_loader, args, device)
        
        if acc > best_acc:
            best_acc = acc
            best_config = (emb_dim, hid_dim, lr, epochs)
            best_model = model
        
        index += 1
    
    print(best_config)
    torch.save(best_model.state_dict(), args.model_path)
    print(f'Model saved to {args.model_path}')
    

    #if you need to the load the model after it's trained
    # embedding_dim = 96
    # hidden_dim = 192

    # print('Loading best model...')
    # model = SyllableRNN(
    #     vocab_size=len(graph_vocab),
    #     tagset_size=len(tag_vocab),
    #     embedding_dim = embedding_dim,
    #     hidden_dim = hidden_dim,
    # ).to(device)
    # model.load_state_dict(torch.load(args.model_path))
                               

    print('Evaluating on test set...')
    test_acc, (all_preds, all_targets) = evaluate(best_model, test_loader, device)
    print(f'Test Accuracy: {test_acc:.4f}')
    
    print('Generating confusion matrix...')
    conf_matrix = generate_confusion_matrix(best_model, test_loader, tag_vocab, device)
    plot_confusion_matrix(conf_matrix, tag_vocab)
    print('Confusion matrix saved to confusion_matrix.png')

    with open("GRU/details.txt", "w") as d:
        d.write("Paramters: " + ", ".join(map(str, best_config)) + "\n")
        d.write("Test Accuracy: " + str(test_acc)+ '\n')
        d.write("Confusion Matrix:  \n")
        d.write(" ".join(map(str,conf_matrix[0])) + '\n')
        d.write(" ".join(map(str,conf_matrix[1])) + '\n')
    
    d.close()
    
    write_all_output(best_model, test_loader, graph_vocab, tag_vocab, device)
    write_all_errors(best_model, test_loader, graph_vocab, tag_vocab, device)


if __name__ == "__main__":
    main()