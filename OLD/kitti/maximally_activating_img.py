
#%%
import torch
from torch import nn
from torch.autograd import Variable
import arch

MAIN = __name__ == "__main__"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
#load data
clean_model = arch.MNIST_Net()
poison_model = arch.MNIST_Net()

clean_dict = torch.load(open("../../models/clean/clean_0004149_4.pt", "rb"))
poison_dict = torch.load(open("../../models/poison/poison_0004149_4.pt", "rb"))
clean_model.load_state_dict(clean_dict)
poison_model.load_state_dict(poison_dict)

def generate_image(net, neuron_idx, num_iterations=1000, step_size=1.0):
    # Initialize the image to random noise
    image = Variable(torch.randn(1, 1, 28, 28), requires_grad=True)

    # Define the loss function as the negative activation of the chosen neuron
    loss_fn = nn.MSELoss(reduction='mean')

    for i in range(num_iterations):
        # Compute the forward pass of the network
        output = net(image)

        # Compute the loss as the negative activation of the chosen neuron
        loss = -output[0, neuron_idx]

        # Zero out the gradients of the image
        net.zero_grad()
        image.grad = None

        # Compute the gradients of the loss with respect to the image
        loss.backward()

        # Update the image using gradient ascent
        image.data += step_size * image.grad.data

        # Clip the image to keep its pixel values between 0 and 1
        image.data = torch.clamp(image.data, 0, 1)

    # Return the final image as a numpy array
    return image.data.numpy()[0, 0, :, :]


net = arch.MNIST_Net()
image = generate_image(net, 3)

#%%
plt.imshow(image)