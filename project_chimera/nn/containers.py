# project_chimera/nn/containers.py

from project_chimera.nn.module import Module

class Sequential(Module):
    """
    A sequential container for modules.
    Modules will be added to it in the order they are passed in the constructor.
    """
    def __init__(self, *args):
        super().__init__()
        self.modules = []
        for i, module in enumerate(args):
            # To avoid name conflicts, we use an internal list
            self.modules.append(module)
            # This setattr is important for the .parameters() method to find the submodules
            setattr(self, f"module_{i}", module)

    def forward(self, input_tensor):
        """
        Passes the input through all modules in order.
        """
        current_tensor = input_tensor
        for module in self.modules:
            current_tensor = module(current_tensor)
        return current_tensor

    def __repr__(self):
        return "Sequential(\n" + ",\n".join([f"  {m}" for m in self.modules]) + "\n)"
