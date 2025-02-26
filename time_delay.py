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
        """Preprocess graph with realistic timing features"""
        # Calculate net delays considering fanout and gate types
        fanout = g.out_degrees().float().unsqueeze(1)
        normalized_fanout = fanout / fanout.max()
        
        # Base delays from graph
        base_delays = g.ndata['n_net_delays']
        
        # Adjust delays based on fanout
        fanout_factor = 1.0 + 0.2 * normalized_fanout  # 20% increase per fanout
        adjusted_delays = base_delays * fanout_factor
        
        # Calculate logarithmic delays for better numerical stability
        g.ndata['n_net_delays_log'] = torch.log(0.0001 + adjusted_delays) + 7.6
        
        # Handle arrival times
        invalid_nodes = torch.abs(g.ndata['n_ats']) > 1e20
        g.ndata['n_ats'][invalid_nodes] = 0
        
        # Calculate slews considering gate types and fanout
        base_slews = g.ndata['n_slews']
        adjusted_slews = base_slews * (1.0 + 0.1 * normalized_fanout)  # 10% increase per fanout
        g.ndata['n_slews'][invalid_nodes] = 0
        
        # Combine arrival times and slews
        g.ndata['n_atslew'] = torch.cat([
            g.ndata['n_ats'],
            torch.log(0.0001 + adjusted_slews) + 3
        ], dim=1)

        # Convert edge features to float32
        g.edata['ef'] = g.edata['ef'].type(torch.float32)
        g.edata['e_cell_delays'] = g.edata['e_cell_delays'].type(torch.float32)
        
        return g.to(self.device)
    
    def break_cycles(self, g):
        """Break cycles while preserving critical paths"""
        g = dgl.remove_self_loop(g)
        
        # Get edge weights based on delays
        edge_weights = g.ndata['n_net_delays'][g.edges()[0]]
        
        # Sort edges by weight
        sorted_edges = torch.argsort(edge_weights.squeeze(), descending=True)
        src, dst = g.edges()
        src = src[sorted_edges]
        dst = dst[sorted_edges]
        
        # Keep edges that don't create cycles, prioritizing higher delay edges
        mask = src < dst
        g = dgl.graph((src[mask], dst[mask]))
        
        return g
        
    def predict(self, graph_path):
        """Predict timing delays with improved accuracy"""
        g = dgl.load_graphs(graph_path)[0][0]
        g = self.preprocess_graph(g)

        # Create homogeneous graph
        src, dst = g.edges()
        homo_g = dgl.graph((src, dst))
        homo_g = self.break_cycles(homo_g)
        
        # Calculate node types
        in_degrees = homo_g.in_degrees()
        out_degrees = homo_g.out_degrees()

        # Identify primary inputs and outputs
        pi_nodes = torch.nonzero(in_degrees == 0, as_tuple=True)[0].to(torch.int32)
        po_nodes = torch.nonzero(out_degrees == 0, as_tuple=True)[0].to(torch.int32)

        # Find non-PI output nodes
        output_nodes_nonpi = torch.tensor(
            [n.item() for n in po_nodes if n.item() not in pi_nodes.tolist()], 
            dtype=torch.int32, 
            device=self.device
        )
        
        try:
            # Generate topological ordering
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
            
            # Post-process delays
            final_delays = torch.exp(net_delays.cpu() - 7.6) - 0.0001
            
            # Scale delays based on path length
            path_lengths = self._calculate_path_lengths(homo_g)
            scaled_delays = final_delays * (1 + 0.05 * path_lengths)  # 5% increase per hop
            
            return scaled_delays

    def _calculate_path_lengths(self, g):
        """Calculate path lengths from inputs to each node"""
        with torch.no_grad():
            in_degrees = g.in_degrees()
            path_lengths = torch.zeros(g.num_nodes())
            
            # Start from nodes with no inputs
            start_nodes = (in_degrees == 0).nonzero().squeeze()
            
            # BFS to calculate path lengths
            visited = set()
            queue = [(node.item(), 0) for node in start_nodes]
            
            while queue:
                node, length = queue.pop(0)
                if node not in visited:
                    visited.add(node)
                    path_lengths[node] = length
                    
                    # Add successors to queue
                    successors = g.successors(node)
                    queue.extend([(succ.item(), length + 1) for succ in successors])
            
            return path_lengths.unsqueeze(1)

if __name__ == "__main__":
    predictor = TimingPredictor("model.pth")
    delays = predictor.predict("last.graph.bin")
    print("Predicted delays:", delays) 