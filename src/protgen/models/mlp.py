from protgen.data import ProteinDataset
from torch.utils.data import DataLoader

train_data = ProteinDataset(path_to_data="data/train.csv")
val_data = ProteinDataset(path_to_data="data/val.csv")
test_data = ProteinDataset(path_to_data="data/test.csv")

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

from protgen.data import AA, FAMILIES, IGNORE_INDEX, special_tokens

sequence = train_data.df["sequence"][0]
tokens = train_data.encode(sequence)

import torch
import torch.nn.functional as F

x, x_shifted, y, length = next(iter(train_loader))

# Generative: one distribution over the vocabulary per position.
logits_gen = torch.randn(x.shape[0], train_data.vocab_size, train_data.max_length)

# Classification: one distribution over the three families per sequence.
logits_fam = torch.randn(x.shape[0], len(FAMILIES))

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

hidden_layer_nodes = 50
#num_features = 3 # The three vakues in the sting of protein
num_features = train_data.vocab_size # The three vakues in the sting of protein, but this have I made as a array that is onehot encoded
num_output = train_data.vocab_size # The number of of possible values
num_embedding = 128 # The number of embedding dimensions

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Setting up variables, these variables are weights in your 
        # network that can be updated while running our graph.
        # Notice, to make a hidden layer, the weights need to have the 
        # following dimensionality:
        #   W[number_of_units_going_out, number_of_units_going_in]
        #   b[number_of_units_going_out]
        # in the example below we have 2 input units (num_features) and 2 output units (num_output)
        # so our weights become W[2, 2], b[2]
        # if we want to make a hidden layer with 100 units, we need to define the shape of the
        # first weight to W[100, 2], b[2] and the shape of the second weight to W[2, 100], b[2]
        
        # first layer
        #self.W_1 = nn.Parameter(torch.randn(num_output, num_features)) 
        #self.b_1 = nn.Parameter(torch.randn(num_output))
        self.embed = nn.Embedding(num_features, num_embedding)
        self.W_1 = nn.Parameter(torch.randn(hidden_layer_nodes, num_embedding*3)) 
        self.b_1 = nn.Parameter(torch.randn(hidden_layer_nodes))
            
        # second layer (to be completed as an exercise)
        # NB when you create a second layer, remember that you also must change parts of the first layer
        self.W_2 = nn.Parameter(torch.randn(hidden_layer_nodes, hidden_layer_nodes))
        self.b_2 = nn.Parameter(torch.randn(hidden_layer_nodes))

        self.W_3 = nn.Parameter(torch.randn(num_output, hidden_layer_nodes))
        self.b_3 = nn.Parameter(torch.randn(num_output))


        # He initializatopm fp weight followinf taking inspiration from figure 7.8 in Prince
        # Without having this, the exploding gradient problem ruined the training process (see Prince section 7.5)
        nn.init.kaiming_uniform_(self.W_1, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.W_2, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.W_3, nonlinearity="relu")

        nn.init.zeros_(self.b_1)
        nn.init.zeros_(self.b_2)
        nn.init.zeros_(self.b_3)
        
    def forward(self, x):
        # Setting up ops, these ops will define edges along our computational graph
        # The below ops will compute a logistic regression, 
        # but can be modified to compute a neural network
        #print(f"shape of x before embedding: {x.shape}")
        #print(x[0, :])
        x = self.embed(x) 
        #print(f"shape of x after embedding: {x.shape}")
        #print(x[0, :, :])
        x = x.flatten(start_dim=1)
        #print(f"shape of x after flatten: {x.shape}")
        #print(x[0, :])
        x = F.linear(x, self.W_1, self.b_1) # være rigtige dimensioner 
        #print(f"shape of x after 1 hidden layer: {x.shape}")
        #print(x[0, :])
        #print("Weight W1: ", self.W_1)
        #print("Weight b1: ", self.b_1)
        
        # second layer (to be completed as an exercise)
        # NB when you create a second layer, remember that you also must change parts of the first layer
        #x = F.linear(x, self.W_2, self.b_2)
        x = F.relu(x)
        x = F.linear(x, self.W_2, self.b_2)

        x = F.relu(x)
        x = F.linear(x, self.W_3, self.b_3)

        return F.softmax(x, dim=1) # x # F.softmax(x, dim=1) # softmax to be performed on the second dimension

net = Net()

def cross_entropy(ys, ts):
    # computing cross entropy per sample
    cross_entropy = -torch.sum(ts * torch.log(ys), dim=1, keepdim=False)
    # averaging over samples
    return torch.mean(cross_entropy)

import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=1., momentum=0.9)

#def accuracy(ys, ts):
#    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
#    correct_prediction = torch.eq(torch.max(ys, 1)[1], torch.max(ts, 1)[1])
#    # averaging the one-hot encoded vector
#    return torch.mean(correct_prediction.float())
def accuracy(ys, ts):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    correct_prediction = torch.eq(torch.max(ys, 1)[1], ts) # tr_targets is the last value and there there is not an extra index dimension
    # averaging the one-hot encoded vector
    return torch.mean(correct_prediction.float())


batch_size_ = 32*4
train_loader = DataLoader(train_data, batch_size=batch_size_, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size_, shuffle=True)

size_of_traindata = len(train_data)

def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out

# number of training passses
num_epochs = int(np.floor(size_of_traindata/batch_size_))
num_epochs = 500
#num_epochs = 
# store loss and accuracy for information
train_losses, val_losses, train_accs, val_accs = [], [], [], []

def pred(X):
    """ Compute graph's prediction and return numpy array
    
    Parameters
    ----------
    X : numpy.ndarray
    
    Returns
    -------
    numpy.ndarray
    """
    X = torch.from_numpy(X)
    y = net(X)
    return y.data.numpy()

# training loop
for e in range(num_epochs):
    x, x_shifted, y, length = next(iter(train_loader))
    x = F.pad(x, (2, 0, 0, 0)) # Zero padding with 2 columns to the left.
    for i in range(x.shape[0]):
        progress = 0
        endOfString = 0
        while endOfString != 2:
            if (i == 0) and (progress == 0):
                X_tr = x[i, 0:3]
                y_tr = x_shifted[i, 0]
            else:
                X_tr = np.vstack((X_tr, x[i, (0+progress):(3+progress)]))
                y_tr = np.vstack((y_tr, x_shifted[i, (0+progress)]))
            endOfString = x_shifted[i, (0+progress)]
            progress += 1

    # get training input and expected output as torch Variables and make sure type is correct
    tr_input = torch.from_numpy(X_tr) # torch.from_numpy(onehot(X_tr, num_output)).float() #X_tr#torch.from_numpy(onehot(X_tr, num_output)).float()# torch.from_numpy(X_tr)
    tr_targets = torch.from_numpy(y_tr) #torch.from_numpy(onehot(y_tr, num_output)).float() #y_tr#torch.from_numpy(onehot(y_tr, num_output)).float()
    
    # zeroize accumulated gradients in parameters
    optimizer.zero_grad()
    # predict by running forward pass
    #print(X_tr.shape)
    #print(tr_input.shape)
    tr_output = net(tr_input)
    # compute cross entropy loss
    #tr_targetsOutput = net.embed(tr_targets)
    tr_targets_oneHot = torch.Tensor(onehot(tr_targets, num_output)) # Make it one hot such that the code knows which one is the correct one for the cross entropy calculation.
    tr_loss = cross_entropy(tr_output, tr_targets_oneHot) #F.cross_entropy(tr_output, tr_targets.flatten()) #cross_entropy(tr_output, tr_targets)
    # compute gradients given loss
    tr_loss.backward()
    # update the parameters given the computed gradients
    optimizer.step()
    train_acc = accuracy(tr_output, tr_targets)
    
    # store training loss
    train_losses.append(tr_loss.data.numpy())
    train_accs.append(train_acc)

    x_, x_shifted_, y_, length_ = next(iter(val_loader))
    x_ = F.pad(x_, (2, 0, 0, 0)) # Zero padding with 2 columns to the left.
    for i in range(x_.shape[0]):
        progress = 0
        endOfString = 0
        while endOfString != 2:
            if (i == 0) and (progress == 0):
                X_val = x_[i, 0:3]
                y_val = x_shifted_[i, 0]
            else:
                X_val = np.vstack((X_val, x_[i, (0+progress):(3+progress)]))
                y_val = np.vstack((y_val, x_shifted_[i, (0+progress)]))
            endOfString = x_shifted_[i, (0+progress)]
            progress += 1
    # get validation input and expected output as torch Variables and make sure type is correct
    val_input = torch.from_numpy(X_val) # torch.from_numpy(onehot(X_val, num_output)).float()  #torch.from_numpy(X_val)
    val_targets = torch.from_numpy(y_val)  #torch.from_numpy(onehot(y_val, num_output)).float()
    
    # predict with validation input
    val_output = net(val_input)
    # compute loss and accuracy
    val_targets_oneHot = torch.Tensor(onehot(val_targets, num_output)) # Make it one hot such that the code knows which one is the correct one for the cross entropy calculation.
    val_loss = cross_entropy(val_output, val_targets_oneHot) #F.cross_entropy(val_output, val_targets.flatten()) #cross_entropy(val_output, val_targets)
    #val_loss = cross_entropy(val_output, val_targets)
    val_acc = accuracy(val_output, val_targets)
    
    # store loss and accuracy
    val_losses.append(val_loss.data.numpy())
    val_accs.append(val_acc.data.numpy())
    
    if e % 10 == 0:
        print("Epoch %i, "
              "Train Cost: %0.3f"
              "\tVal Cost: %0.3f"
              "\t Val acc: %0.3f" % (e, 
                                     train_losses[-1],
                                     val_losses[-1],
                                     val_accs[-1]))
        
        
## get test input and expected output
#te_input = torch.from_numpy(X_te)
#te_targets = torch.from_numpy(onehot(y_te, num_output)).float()
## predict on testset
#te_output = net(te_input)
## compute loss and accuracy
#te_loss = cross_entropy(te_output, te_targets)
#te_acc = accuracy(te_output, te_targets)
#print("\nTest Cost: %0.3f\tTest Accuracy: %0.3f" % (te_loss.data.numpy(), te_acc.data.numpy()))

# plt.figure()
# epoch = np.arange(len(train_losses))
# plt.plot(epoch, train_losses, 'r', label='Train Loss')
# plt.plot(epoch, val_losses, 'b', label='Val Loss')
# plt.legend()
# plt.xlabel('Updates')
# plt.ylabel('Loss')
# plt.show()

# plt.figure()
# plt.plot(epoch, train_accs, 'r', label='Train Acc')
# plt.plot(epoch, val_accs, 'b', label='Val Acc')
# plt.legend()
# plt.xlabel('Updates')
# plt.ylabel('Accuracy')
# plt.show()

# Create reverse mapping: index -> token
idx_to_token = {v: k for k, v in train_data.vocab.items()}

# With trained model, generate seqeunces:
max_length=129
ole_done_done = [[]]
for j in range(100): # Generate 10 sequences
    ole = torch.Tensor([[0,0,1]]).int()
    ole_done = [1]
    for i in range(max_length-1):
        tr_output = net(ole)
        # next_token = tr_output.argmax() NOT ARGMAX but a smapling over the dsitribution. Otherwise we would always generate the same string! We need this wee bit of "noise" to make this a generative model that makes snese.
        # Make sure that it does not take a padding or <begin of string> token:
        GoodToken = True
        while GoodToken:
            next_token = torch.multinomial(tr_output, num_samples=1)
            if next_token.item() != 0 and next_token.item() != 1:
                GoodToken = False
        #next_token = torch.multinomial(tr_output, num_samples=1)
        if next_token.item() == 2: # If the next token is the end of sequence token, then we stop generating.
                    break
        ole = torch.cat(
            (ole[:, 1:], next_token),
            dim=1
        )
        ole_done.append(next_token[0].item())
        # if next_token.item() == 2: # If the next token is the end of sequence token, then we stop generating.
        #     break
    #print(f"Generated sequence {j+1}: {ole_done}")
    ole_done = ole_done[1:] # remove beging of string token
    tokens = [idx_to_token[i] for i in ole_done]
    ole_done_done.append("".join(tokens))
ole_done_done = ole_done_done[1:] # Remove the first black sequence this variable was initalised with.

# Make the CSV file:
import csv

with open('results/sequences/generated_sequences_MLP.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['sequence'])
    for seq in ole_done_done:
        writer.writerow([seq])

