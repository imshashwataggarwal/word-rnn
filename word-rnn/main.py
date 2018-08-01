import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import *
import numpy as np
import os

path = './data'
script_name = 'harrypotter.txt'

# Hyper Parameters
seq_len = 32
batch_size = 16
embed_size = 128
hidden_size = 128
num_layers = 2
epochs = 50
learning_rate = 2e-3
grad_clip = 0.5
sample_len = 200

# Data Pipeline
print("Creating Training Data")

train_path = os.path.join(path,script_name)
test_path = os.path.join(path,'test.txt')

corpus = Corpus()
int_text = corpus.get_data(train_path, batch_size)

vocab_size = len(corpus.dictionary)
num_batches = int_text.size(1) // seq_len

print("Data Loaded!")

print("Initializing Model!")
# Model
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, h):
        x = self.embed(x)
        out, h = self.lstm(x, h)
        out = out.contiguous().view(out.size(0) * out.size(1), out.size(2))
        out = self.linear(out)
        return out, h

# Loss and Optimizer
print("Setting up loss and optimizer")
rnn = RNN(vocab_size, embed_size, hidden_size, num_layers).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr = learning_rate)
#scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)


# Training
print("Training Started ...")
for epoch in range(epochs):

    # Initial hidden and memory states
    states = (Variable(torch.zeros(num_layers, batch_size, hidden_size)).cuda(), Variable(torch.zeros(num_layers, batch_size, hidden_size)).cuda())

    for i in range(0, int_text.size(1) - seq_len, seq_len):

        inputs = Variable(int_text[:, i:i+seq_len]).cuda()
        targets = Variable(int_text[:, (i+1):(i+1)+seq_len].contiguous()).cuda()

        # Forward + Backward + Optimize
        rnn.zero_grad()
        outputs, states = rnn(inputs, states)
        loss = criterion(outputs, targets.view(-1))
        #scheduler.step(loss)
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm(rnn.parameters(), grad_clip)
        optimizer.step()

        step = (i+1) // seq_len
        if step % 100 == 0:
            print ('Epoch [%d/%d], Step[%d/%d], Loss: %.3f, Perplexity: %5.2f' %
                   (epoch+1, epochs, step, num_batches, loss.data[0], np.exp(loss.data[0])))

    if (epoch + 1)%5 == 0:
	    torch.save(rnn.state_dict(), 'rnn-' + str(epoch + 1) + '.pkl')
# Save the Trained Model
print("Saving Model ...")
torch.save(rnn.state_dict(), 'rnn.pkl')

# Sampling
# TODO: Implement Beam Search and Inference Scoring!
print("Testing Model ...")
rev_puncts = {v:k for k,v in puncts.items()}
rev_contractions = {v:k for k,v in contractions.items()}

with open(test_path, 'w') as f:
    # Set intial hidden ane memory states
    state = (Variable(torch.zeros(num_layers, 1, hidden_size)).cuda(), Variable(torch.zeros(num_layers, 1, hidden_size)).cuda())

    # Select one word id randomly
    prob = torch.ones(vocab_size)
    inputs = Variable(torch.multinomial(prob, num_samples=1).unsqueeze(1), volatile=True).cuda()

    text_prin = ""
    for i in range(sample_len):
        # Forward propagate rnn
        output, state = rnn(inputs, state)

        # Sample a word id
        prob = output.squeeze().data.exp().cpu()
        word_id = torch.multinomial(prob, 1)[0]

        # Feed sampled word id to next time step
        inputs.data.fill_(word_id)

        # File write
        word = corpus.dictionary.idx2word[word_id]
        flag = 0

        if word in rev_puncts:
            word = rev_puncts[word]
            if not word in ['"','(',]:
                flag = 1
            if not word in ['"','(','--','-','—','\n']:
                word = word + ' '

        elif word in rev_contractions:
            word = rev_contractions[word] + ' '

        elif word == '<EOS>':
            word = '\n'

        else:
            if word[0] == '`' or word[0] == '’':
                flag = 1
            word = word + ' '

        if len(text_prin) > 0:
            if flag and text_prin[-1] == " ":
                text_prin = text_prin[:-1] + word
            else:
                text_prin = text_prin + word
        else:
            text_prin = text_prin + word

        if (i+1) % 100 == 0:
            print('Sampled [%d/%d] words and save to %s'%(i+1, num_samples, sample_path))

    # Output to file
    f.write(text_prin)
    print(text_prin)
