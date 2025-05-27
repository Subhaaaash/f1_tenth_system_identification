import torch
import torch.nn as nn

class MemoryNeuralNetwork(nn.Module):
    def __init__(self,
                 number_of_input_neurons=5,
                 number_of_hidden_neurons=100,
                 number_of_output_neurons=3,
                 neeta=4e-5,
                 neeta_dash=4e-5,
                 lipschitz_norm=1.0,
                 spectral_norm=False,
                 seed_value=16981):
        super(MemoryNeuralNetwork, self).__init__()
        torch.manual_seed(seed_value)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # dimensions
        self.I = number_of_input_neurons
        self.H = number_of_hidden_neurons
        self.O = number_of_output_neurons

        # learning rates
        self.neeta = neeta
        self.neeta_dash = neeta_dash

        # spectral norm
        self.spectral_norm = spectral_norm
        self.lipschitz = lipschitz_norm

        # memory coefficients
        self.alpha_input_layer  = nn.Parameter(torch.rand(self.I, device=self.device))
        self.alpha_hidden_layer = nn.Parameter(torch.rand(self.H, device=self.device))
        self.alpha_last_layer   = nn.Parameter(torch.rand(self.O, device=self.device))

        # weights
        self.beta = nn.Parameter(torch.rand(self.O, device=self.device))
        self.W_ih_nn = nn.Parameter(torch.rand(self.I, self.H, device=self.device))
        self.W_ho_nn = nn.Parameter(torch.rand(self.H, self.O, device=self.device))
        self.W_ih_mn = nn.Parameter(torch.rand(self.I, self.H, device=self.device))
        self.W_ho_mn = nn.Parameter(torch.rand(self.H, self.O, device=self.device))

        # past-state buffers
        self.prev_x_nn = torch.zeros(self.I, device=self.device)
        self.prev_x_mn = torch.zeros(self.I, device=self.device)
        self.prev_h_nn = torch.zeros(self.H, device=self.device)
        self.prev_h_mn = torch.zeros(self.H, device=self.device)
        self.prev_o_nn = torch.zeros(self.O, device=self.device)
        self.prev_o_mn = torch.zeros(self.O, device=self.device)

        self.to(self.device)

    def feedforward(self, x_in):
        """
        x_in: a length-I numpy array or tensor of inputs at current time
        returns: length-O tensor
        """
        # ensure tensor on correct device
        x = torch.tensor(x_in, dtype=torch.float32, device=self.device)

        # memory neuron at input layer
        x_mem = self.alpha_input_layer * self.prev_x_nn + (1 - self.alpha_input_layer) * self.prev_x_mn

        # hidden layer
        h_in_nn = (self.W_ih_nn.t() @ x) + (self.W_ih_mn.t() @ x_mem)
        h_nn    = 15 * torch.tanh(h_in_nn / 15)
        h_mem   = self.alpha_hidden_layer * self.prev_h_nn + (1 - self.alpha_hidden_layer) * self.prev_h_mn

        # memory neuron at output layer
        o_mem = self.alpha_last_layer * self.prev_o_nn + (1 - self.alpha_last_layer) * self.prev_o_mn

        # output layer
        o_in = (self.W_ho_nn.t() @ h_nn) + (self.W_ho_mn.t() @ h_mem) + self.beta * o_mem
        o_nn = o_in  # identity activation on output

        # update past states
        self.prev_x_nn = x.clone()
        self.prev_x_mn = x_mem.clone()
        self.prev_h_nn = h_nn.clone()
        self.prev_h_mn = h_mem.clone()
        self.prev_o_nn = o_nn.clone()
        self.prev_o_mn = o_mem.clone()

        # store latest output for rmse
        self.output = o_nn
        return o_nn

    def backprop(self, y_des):
        """
        y_des: length-O numpy array or tensor of desired outputs
        Updates weights & memory coefficients in-place.
        """
        y = torch.tensor(y_des, dtype=torch.float32, device=self.device)
        # error at output
        e_o = (self.output - y)  # derivative of identity is 1

        # error back to hidden (only via NN weights)
        de_dh = self.W_ho_nn @ e_o
        dh = (1 - torch.tanh(self.prev_h_nn / 15).pow(2))  # derivative of 15*tanh(x/15)
        e_h = de_dh * dh

        # update NN weights
        # outer products: [I x H] gets gradient ∝ x * e_h
        self.W_ih_nn.data -= self.neeta * torch.ger(self.prev_x_nn, e_h)
        self.W_ih_mn.data -= self.neeta * torch.ger(self.prev_x_mn, e_h)
        # [H x O] gets gradient ∝ h_nn * e_o
        self.W_ho_nn.data -= self.neeta * torch.ger(self.prev_h_nn, e_o)
        self.W_ho_mn.data -= self.neeta * torch.ger(self.prev_h_mn, e_o)
        self.beta.data    -= self.neeta_dash * (e_o * self.prev_o_mn)

        # memory grads
        pd_e_v_h = self.W_ho_mn @ e_o
        pd_e_v_x = self.W_ih_mn @ e_h
        pd_e_v_o = self.beta * e_o

        pd_v_alpha_h = self.prev_h_nn - self.prev_h_mn
        pd_v_alpha_x = self.prev_x_nn - self.prev_x_mn
        pd_v_alpha_o = self.prev_o_nn - self.prev_o_mn

        self.alpha_hidden_layer.data -= self.neeta_dash * pd_e_v_h * pd_v_alpha_h
        self.alpha_input_layer.data  -= self.neeta_dash * pd_e_v_x * pd_v_alpha_x
        self.alpha_last_layer.data   -= self.neeta_dash * pd_e_v_o * pd_v_alpha_o

        # clamp to [0,1]
        self.alpha_hidden_layer.data.clamp_(0,1)
        self.alpha_input_layer.data.clamp_(0,1)
        self.alpha_last_layer.data.clamp_(0,1)
        self.beta.data.clamp_(0,1)
