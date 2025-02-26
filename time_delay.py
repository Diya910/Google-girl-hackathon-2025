import torch
import dgl
import numpy as np
from gnn_model import TimingGCN

class TimingPredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = TimingGCN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def preprocess_graph(self, g):
        """Preprocess graph for a homogeneous graph"""
        g.ndata['n_net_delays_log'] = torch.log(0.0001 + g.ndata['n_net_delays']) + 7.6
        invalid_nodes = torch.abs(g.ndata['n_ats']) > 1e20
        g.ndata['n_ats'][invalid_nodes] = 0
        g.ndata['n_slews'][invalid_nodes] = 0
        g.ndata['n_atslew'] = torch.cat([
            g.ndata['n_ats'],
            torch.log(0.0001 + g.ndata['n_slews']) + 3
        ], dim=1)

        g.edata['ef'] = g.edata['ef'].type(torch.float32)
        g.edata['e_cell_delays'] = g.edata['e_cell_delays'].type(torch.float32)
        
        return g.to(self.device)
    
    def break_cycles(self, g):
        g = dgl.remove_self_loop(g)
        src, dst = g.edges()
        mask = src < dst
        g = dgl.graph((src[mask], dst[mask]))
        
        return g
        
    def predict(self, graph_path):
        g = dgl.load_graphs(graph_path)[0][0]
        g = self.preprocess_graph(g)

        src, dst = g.edges()
        homo_g = dgl.graph((src, dst))
        homo_g = self.break_cycles(homo_g)
        
        in_degrees = homo_g.in_degrees()
        out_degrees = homo_g.out_degrees()

        pi_nodes = torch.nonzero(in_degrees == 0, as_tuple=True)[0].to(torch.int32)
        po_nodes = torch.nonzero(out_degrees == 0, as_tuple=True)[0].to(torch.int32)

        output_nodes_nonpi = torch.tensor(
            [n.item() for n in po_nodes if n.item() not in pi_nodes.tolist()], 
            dtype=torch.int32, 
            device=self.device
        )
        
        try:
            topo_nodes_raw = list(dgl.topological_nodes_generator(homo_g))
            topo_nodes = [nodes.to(torch.int32) for nodes in topo_nodes_raw]
        except dgl._ffi.base.DGLError:
            print("Warning: Cycles detected in graph. Using node IDs as fallback.")
            all_nodes = torch.arange(homo_g.num_nodes(), dtype=torch.int32, device=self.device)
            chunk_size = 100
            topo_nodes = [all_nodes[i:i+chunk_size] for i in range(0, len(all_nodes), chunk_size)]

        if len(topo_nodes) % 2 != 0:
            topo_nodes.append(torch.tensor([], dtype=torch.int32, device=self.device))
       
        topo_dict = {
            "topo": topo_nodes,
            "pi_nodes": pi_nodes,
            "input_nodes": pi_nodes if len(pi_nodes) > 0 else torch.tensor([0], dtype=torch.int32, device=self.device),
            "output_nodes": po_nodes,
            "output_nodes_nonpi": output_nodes_nonpi
        }
        
        with torch.no_grad():
            net_delays, cell_delays, _ = self.model(g, topo_dict)

        return torch.exp(net_delays.cpu() - 7.6) - 0.0001

if __name__ == "__main__":
    predictor = TimingPredictor("/home/diya/Girl_hackathon/Google-girl-hackathon-2025/model.pth")
    delays = predictor.predict("/home/diya/Girl_hackathon/Google-girl-hackathon-2025/last.graph.bin")
    print("Predicted delays:", delays)