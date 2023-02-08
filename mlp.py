from torch import nn
import yaml




class MLP(nn.Module):
    def __init__(self, yaml_path):
        super(MLP, self).__init__()
        self.yaml_path = yaml_path
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)

        self.device = config["device"]
        all_layers = []
        for layer_yaml in config["layers"]:
            layer = []
            if "dropout" in config["layers"][layer_yaml]:
                layer.append(nn.Dropout(config["layers"][layer_yaml]["dropout"]))
            if "linear" in config["layers"][layer_yaml]:
                layer.append(nn.Linear(config["layers"][layer_yaml]["linear"]["in_dim"], config["layers"][layer_yaml]["linear"]["out_dim"], device=self.device))
            if "activation" in config["layers"][layer_yaml]:
                layer.append(nn.LeakyReLU())

            if layer:
                all_layers.append(nn.Sequential(*layer))

        self.forward_pass = nn.Sequential(*all_layers)

    def forward(self, x):
        output = self.forward_pass(x)
        return output
