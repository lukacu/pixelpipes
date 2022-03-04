import torch
import torchvision

torch.cuda.init()

device = torch.device("cuda:0")
device = torch.device("cpu")

"""
PREPARE TRAIN SET AND TEST SET
"""

image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128,128)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5], inplace = True)
])


dataset = torchvision.datasets.MNIST(
    "",
    train = True,
    download = True,
    transform = image_transform)
    
data = torch.utils.data.DataLoader(
    dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    pin_memory = True)
    
"""
NEURAL NETWORK ARHITECTURE
"""

class NeuralNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()      
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, stride=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(64, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1)
        )
        
        self.decoder = torch.nn.Sequential(    
            torch.nn.ConvTranspose2d(32, 64, 4, stride=2),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64, 32, 5, stride=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(32, 1, 2, stride=2, padding=2),
            torch.nn.Tanh()   
        )
        
    def forward(self, input):  
        encoder_out = self.encoder(input)
        decoder_out = self.decoder(encoder_out)
        return decoder_out
 
neural_network = NeuralNetwork().to(device)

"""
NEURAL NETWORK TRAINING
"""

EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.00001

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    neural_network.parameters(),
    lr = LEARNING_RATE, 
    weight_decay = WEIGHT_DECAY)

loss_list = []

for epoch in range(EPOCHS):

    for batch in data:
        input_image, _ = batch
        
        input_image = input_image.to(device)
    
        # Run the forward pass
        output_image = neural_network(input_image)
        
        #break
        loss = criterion(output_image, input_image)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print("epoch [{}/{}], loss:{:.4f}".format(epoch+1, EPOCHS, loss.item()))

# TODO