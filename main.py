import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
words = open("names.txt", "r").read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# Hyperparameters
block_size = 14
encoding_size = 5
hidden_leyer_size = 200

X,Y = [], []
for w in words:
	
	
	context = [0] * block_size
	for ch in w + '.':
		ix = stoi[ch]
		X.append(context)
		Y.append(ix)
		
		context = context[1:] + [ix] # crop and append
	
X = torch.tensor(X)
Y = torch.tensor(Y)

C = torch.randn((27,encoding_size))
W1 = torch.randn((encoding_size * block_size, hidden_leyer_size)) * 0.2
b1 = torch.randn(hidden_leyer_size) * 0
W2 = torch.randn((hidden_leyer_size, 27)) * 0.2
b2 = torch.randn(27) * 0
parameters = [C, W1,b1,W2, b2]

for p in parameters:
	p.requires_grad = True

lri = []
lossi = []
stepi = []

for i in range(100000):

	ix = torch.randint(0, X.shape[0], (100, ))

	emb = C[X[ix]]
	h = torch.tanh(emb.view(-1, encoding_size * block_size) @ W1 + b1)
	logits = h @ W2 + b2
	loss = F.cross_entropy(logits, Y[ix])


	for p in parameters:
		p.grad = None

	loss.backward()

	lr = 0.1 if i < 100000 else 0.01
	for p in parameters:
		p.data += -lr * p.grad

	stepi.append(i)
	lossi.append(loss.log10().item())

plt.plot(stepi, lossi)
plt.show()

for _ in range(200):
    
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      emb = C[torch.tensor([context])] # (1,block_size,d)
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out))
    
emb = C[X]
h = torch.tanh(emb.view(-1, encoding_size * block_size) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y)
print(f"{loss=}")