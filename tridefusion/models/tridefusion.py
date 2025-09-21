import torch


class TriDeFusion:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.build_model()
        self.model.load_state_dict(torch.load(args.model_path, map_location=torch.device(self.device)))
        self.model.eval()
        print("Model preloaded successfully!")
