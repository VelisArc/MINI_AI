# project_chimera/nn/resnet_block.py
from .module import Module
from .conv2d import Conv2d
from .activations import ReLU

class ResnetBlock(Module):
    """
    A simple ResNet block with two convolutional layers.
    (Padding fix के साथ)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # पहला कनवल्शनल ब्लॉक, 3x3 कर्नल के साथ आकार बनाए रखता है
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        
        # --- THE FIX IS HERE ---
        # दूसरा कनवल्शनल ब्लॉक, 1x1 कर्नल के साथ आकार बनाए रखता है
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
        if in_channels != out_channels:
            # 1x1 कर्नल के साथ शॉर्टकट को भी padding=0 चाहिए
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = None

    def forward(self, x):
        h = self.relu1(self.conv1(x))
        h = self.conv2(h)
        
        if self.shortcut:
            shortcut_x = self.shortcut(x)
        else:
            shortcut_x = x
            
        return h + shortcut_x
