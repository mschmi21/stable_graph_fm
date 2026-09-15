from src.models.gnn import GNN
from src.models.mlp import MLP
from src.models.TSBM.GCNPolicy import GCNPolicy

def get_model(config: dict, L= None):

    name = config["model"]["name"]

    if name == "gnn":
        return GNN(
            L = L, 
            K=config["model"]["K"],
            F_hidden=config["model"]["F_hidden"],
            num_layers=config["model"]["num_layers"]
        )
    
    if name == "mlp":
        return MLP(
            N = config["model"]["N"],
            F_hidden=config["model"]["F_hidden"]
        )
    
    if name == "gcn_policy":
        return GCNPolicy(
            data_dim=config["model"]["N"],
            hidden_dim=config["model"]["F_hidden"],
            time_embed_dim=config["model"]["time_embed_dim"]
        )

    raise ValueError(f"Unknown model: {name}")