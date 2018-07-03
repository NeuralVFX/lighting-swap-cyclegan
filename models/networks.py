import torch.nn as nn

############################################################################
# Re-usable blocks
############################################################################


class ConvTrans(nn.Module):
    # One Block to be used as conv and transpose throughout the model
    def __init__(self, ic=4, oc=4, kernel_size=3, block_type='res', padding=1, store_relu=False, stride=2):
        super(ConvTrans, self).__init__()
        self.store_relu = store_relu
        self.block_type = block_type

        if self.block_type == 'up':
            self.conv = nn.ConvTranspose2d(in_channels=ic, out_channels=oc, padding=padding, output_padding=padding,
                                           kernel_size=kernel_size, stride=stride, bias=False)
        elif self.block_type == 'down':
            self.conv = nn.Conv2d(in_channels=ic, out_channels=oc, padding=padding, kernel_size=kernel_size,
                                  stride=stride, bias=False)

        self.relu = nn.LeakyReLU(.2, inplace=True)

        self.bn = nn.InstanceNorm2d(oc)

    def forward(self, x):
        x = self.conv(x)
        relu = self.relu(x)
        x = self.bn(relu)
        if self.store_relu:
            return x, relu
        else:
            return x


class ResBlock(nn.Module):
    # Res Block for the center of the model

    def __init__(self, ic=4, oc=4, kernel_size=3, dropout=.5, use_dropout=False):
        super(ResBlock, self).__init__()

        stride = 1

        operations = []
        operations += [nn.ReflectionPad2d(int(kernel_size / 2))]
        operations += [nn.Conv2d(in_channels=ic, out_channels=oc, padding=0, kernel_size=kernel_size, stride=stride)]
        operations += [nn.LeakyReLU(.2, True)]
        operations += [nn.InstanceNorm2d(oc)]

        if use_dropout:
            operations += [nn.Dropout(dropout)]

        operations += [nn.ReflectionPad2d(int(kernel_size / 2))]
        operations += [nn.Conv2d(in_channels=ic, out_channels=oc, padding=0, kernel_size=kernel_size, stride=stride)]
        operations += [nn.InstanceNorm2d(oc)]

        self.block = nn.Sequential(*operations)

    def forward(self, x):
        return x + self.block(x)


############################################################################
# Generator and Discriminator
############################################################################

class Generator(nn.Module):
    # Generator grown from smallest layer
    def __init__(self, layers=3, filts=1024, kernel_size=3, channels=3, res_layers=3):
        super(Generator, self).__init__()

        # residual core
        operations = [ResBlock(ic=filts, oc=filts, use_dropout=True) for i in
                      range(res_layers)]

        # conv and trans building out from core
        for a in range(layers):
            down = [ConvTrans(ic=int(filts // 2), oc=filts, kernel_size=kernel_size, block_type='down')]
            up = [ConvTrans(ic=filts, oc=int(filts // 2), kernel_size=kernel_size, block_type='up')]
            operations = down + operations + up
            filts = int(filts // 2)

        # our input and our output
        inp = [nn.ReflectionPad2d(3),
               nn.Conv2d(in_channels=channels, out_channels=filts, padding=0, kernel_size=7, stride=1), nn.LeakyReLU()]
        out = [nn.ReflectionPad2d(3),
               nn.Conv2d(in_channels=filts, out_channels=channels, padding=0, kernel_size=7, stride=1), nn.Tanh()]

        operations = inp + operations + out

        self.model = nn.Sequential(*operations)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    # Discriminator Which Returns Features For Custom Loss
    def __init__(self, channels=3, filts=512, kernel_size=4, layers=5):
        super(Discriminator, self).__init__()

        operations = []
        out_operations = [nn.Conv2d(in_channels=filts, out_channels=1, padding=0, kernel_size=kernel_size, stride=1)]
        out_operations += [nn.Sigmoid()]
        self.out_operations = nn.Sequential(*out_operations)

        # Build up discriminator backwards based on final filter count
        for a in range(layers):
            if a == layers - 1:
                self.input = ConvTrans(ic=channels, oc=filts, kernel_size=kernel_size, store_relu=False,
                                       block_type='down')
            else:
                operations += [ConvTrans(ic=int(filts // 2), oc=filts, kernel_size=kernel_size, store_relu=True,
                                         block_type='down')]
            filts = int(filts // 2)

        # Store conv operations with ModuleList so we can return a relu in forward loop
        operations.reverse()
        self.operations = nn.ModuleList(operations)

    def forward(self, x):
        # Run operations, and return relu activations for loss function
        x = self.input(x)
        relu_list = []
        for conv in self.operations:
            x, relu = conv(x)
            relu_list.append(relu)
        x = self.out_operations(x)
        return x, relu_list
