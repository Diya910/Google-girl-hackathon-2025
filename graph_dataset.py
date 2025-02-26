import torch
import dgl
import random
import time

# Set fixed seed for reproducibility
random.seed(8026728)

# List of available design benchmarks
circuit_designs = 'blabla usb_cdc_core BM64 jpeg_encoder salsa20 usbf_device aes128 wbqspiflash aes192 cic_decimator xtea aes256 des spm y_huff aes_cipher picorv32a synth_ram zipdiv genericfir usb'.split()

# Randomly select designs for training
training_designs = random.sample(circuit_designs, 14)
compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_topological_order(heterogeneous_graph):
    """
    Generate topological ordering of nodes in the graph.
    
    Args:
        heterogeneous_graph: The heterogeneous graph to process
        
    Returns:
        tuple: Ordered nodes and computation time
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    
    net_src, net_dst = heterogeneous_graph.edges(etype='net_out', form='uv')
    cell_src, cell_dst = heterogeneous_graph.edges(etype='cell_out', form='uv')
    
    combined_graph = dgl.graph((torch.cat([net_src, cell_src]).cpu(), 
                               torch.cat([net_dst, cell_dst]).cpu()))
    
    topological_nodes = dgl.topological_nodes_generator(combined_graph)
    result = [nodes.to(compute_device) for nodes in topological_nodes]
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    
    return result, end_time - start_time

def create_homogeneous_graph_with_features(heterogeneous_graph):
    """
    Convert heterogeneous graph to homogeneous graph with combined features.
    
    Args:
        heterogeneous_graph: Input heterogeneous graph
        
    Returns:
        dgl.DGLGraph: Homogeneous graph with combined features
    """
    # Extract edges from different types
    net_src, net_dst = heterogeneous_graph.edges(etype='net_out', form='uv')
    cell_src, cell_dst = heterogeneous_graph.edges(etype='cell_out', form='uv')
    
    # Create net edge features
    net_features = torch.cat([
        torch.tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]]).expand(len(net_src), 10).to(compute_device),
        heterogeneous_graph.edges['net_out'].data['ef']
    ], dim=1)
    
    # Create cell edge features
    cell_edge_data = heterogeneous_graph.edges['cell_out'].data['ef'][:, 120:512].reshape(len(cell_src), 2*4, 49)
    cell_features = torch.cat([
        torch.tensor([[1., 0.]]).expand(len(cell_src), 2).to(compute_device),
        torch.mean(cell_edge_data, dim=2),
        torch.zeros(len(cell_src), 2).to(compute_device)
    ], dim=1)
    
    # Create bidirectional graph
    homogeneous_graph = dgl.graph((
        torch.cat([net_src, cell_src, net_dst, cell_dst]),
        torch.cat([net_dst, cell_dst, net_src, cell_src])
    ))
    
    # Copy node features
    homogeneous_graph.ndata['nf'] = heterogeneous_graph.ndata['nf']
    homogeneous_graph.ndata['n_atslew'] = heterogeneous_graph.ndata['n_atslew']
    
    # Combine edge features
    homogeneous_graph.edata['ef'] = torch.cat([net_features, cell_features, -net_features, -cell_features])
    
    return homogeneous_graph

# Load and process all graph data
graph_dataset = {}
for design_name in circuit_designs:
    # Load graph from file
    graph = dgl.load_graphs(f'data/8_rat/{design_name}.graph.bin')[0][0].to('cpu')
    
    # Preprocess net delays with logarithmic transform
    graph.ndata['n_net_delays_log'] = torch.log(0.0001 + graph.ndata['n_net_delays']) + 7.6
    
    # Handle invalid nodes
    invalid_nodes = torch.abs(graph.ndata['n_ats']) > 1e20
    graph.ndata['n_ats'][invalid_nodes] = 0
    graph.ndata['n_slews'][invalid_nodes] = 0
    
    # Combine arrival times and slews
    graph.ndata['n_atslew'] = torch.cat([
        graph.ndata['n_ats'],
        torch.log(0.0001 + graph.ndata['n_slews']) + 3
    ], dim=1)
    
    # Convert data types for consistency
    graph.edges['cell_out'].data['ef'] = graph.edges['cell_out'].data['ef'].type(torch.float32)
    graph.edges['cell_out'].data['e_cell_delays'] = graph.edges['cell_out'].data['e_cell_delays'].type(torch.float32)
    
    # Generate topological ordering
    topo_order, topo_time = generate_topological_order(graph)
    
    # Extract important node sets
    timing_stats = {
        'input_nodes': (graph.ndata['nf'][:, 1] < 0.5).nonzero().flatten().type(torch.int32),
        'output_nodes': (graph.ndata['nf'][:, 1] > 0.5).nonzero().flatten().type(torch.int32),
        'output_nodes_nonpi': torch.logical_and(graph.ndata['nf'][:, 1] > 0.5, 
                                               graph.ndata['nf'][:, 0] < 0.5).nonzero().flatten().type(torch.int32),
        'pi_nodes': torch.logical_and(graph.ndata['nf'][:, 1] > 0.5, 
                                     graph.ndata['nf'][:, 0] > 0.5).nonzero().flatten().type(torch.int32),
        'po_nodes': torch.logical_and(graph.ndata['nf'][:, 1] < 0.5, 
                                     graph.ndata['nf'][:, 0] > 0.5).nonzero().flatten().type(torch.int32),
        'endpoints': (graph.ndata['n_is_timing_endpt'] > 0.5).nonzero().flatten().type(torch.long),
        'topo': topo_order,
        'topo_time': topo_time
    }
    
    # Store processed graph and timing statistics
    graph_dataset[design_name] = graph, timing_stats

# Split into training and testing datasets
data_train = {k: t for k, t in graph_dataset.items() if k in training_designs}
data_test = {k: t for k, t in graph_dataset.items() if k not in training_designs}

if __name__ == '__main__':
    for dataset_type, dataset in [("Training", data_train), ("Testing", data_test)]:
        print(f"=== {dataset_type} Dataset Statistics ===")
        for design_name, (graph, timing_stats) in dataset.items():
            print('\\texttt{{{}}},{},{},{},{},{},{}'.format(
                design_name.replace('_', '\_'), 
                graph.num_nodes(), 
                graph.num_edges('net_out'), 
                graph.num_edges('cell_out'), 
                len(timing_stats['topo']), 
                len(timing_stats['po_nodes']), 
                len(timing_stats['endpoints'])
            ))