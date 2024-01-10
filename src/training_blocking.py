import itertools
from datetime import datetime
import random
import numpy as np
from src.dgl_graph.dataset import PubmedSubgraph
from src.utils.models import *
from src.utils.utility import *
import warnings
from src.utils.config import *
import copy
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SparseAdam
from torch.utils.data import DataLoader
from dgl.nn.pytorch import MetaPath2Vec
from tqdm import tqdm

warnings.filterwarnings("ignore")
random.seed(1234)
np.random.seed(1234)
os.environ["DGLBACKEND"] = "pytorch"
min_loss = np.inf
best_embeddings_model_path = "./log/models/GraphSAGE/GraphSAGE-epoch188.ckpt.pth"
load_saved_model = True
current_date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

dataset = PubmedSubgraph(dataset_name="Pubmed Subgraph",
                         subgraph_base_path="../dataset/processed_pubmed_subgraph",
                         neg_etype='equates',
                         raw_dir="../dataset/pubmed_subgraph/raw",
                         save_dir="../dataset/pubmed_subgraph/processed")

sc = dataset.get_sc()

full_graph = dataset.get_graph()[0]

potentially_equates_graph, colleague_graph, citation_graph, collaboration_graph = dataset.get_node_embeddings_graphs()
node_features = full_graph.ndata["feat"]["author"]

embeddings_model = MetaPath2Vec(full_graph, ['potentially_equates'], window_size=1)
dataloader = DataLoader(torch.arange(full_graph.num_nodes('author')), batch_size=128, shuffle=True, collate_fn=embeddings_model.sample)
optimizer = SparseAdam(embeddings_model.parameters(), lr=0.025)

print("Starting training process")
log_dir = f"./log/models/{embeddings_model.__class__.__name__}/"
train_writer = SummaryWriter(log_dir=f"{log_dir}{current_date}/training")
test_writer = SummaryWriter(log_dir=f"{log_dir}{current_date}/testing")
# start server (in gnn-entity-blocking directory): tensorboard --logdir ./log
counter = EARLY_STOPPING
embeddings_model.train()
for e in range(EPOCHS):
    epoch_loss = 0.0
    for (pos_u, pos_v, neg_v) in dataloader:
        loss = embeddings_model(pos_u, pos_v, neg_v)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_writer.add_scalar("Loss", epoch_loss, e)
    print(f"In epoch {e:03d} - loss: {epoch_loss:.4f}")

    if min_loss >= epoch_loss:
        counter = EARLY_STOPPING
        min_valid_loss = epoch_loss
        best_embeddings_model = copy.deepcopy(embeddings_model)
        best_embeddings_model_path = save_checkpoint(embeddings_model, optimizer, e, log_dir)
    counter -= 1
    if counter <= 0:
        print("Early stopping!")
        break
train_writer.close()



# for i, (pos_u, pos_v, neg_v) in enumerate(tqdm(dataloader)):
#     loss = embeddings_model(pos_u, pos_v, neg_v)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
