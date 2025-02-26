import torch
import dgl

def build_graph(nets, instances):
    net_nodes = {net: idx for idx, net in enumerate(nets)}
    instance_nodes = {inst['instance']: idx for idx, inst in enumerate(instances)}

    num_net_nodes = len(nets)
    num_instance_nodes = len(instances)
    
    print(f"Total nets: {num_net_nodes}, Total instances: {num_instance_nodes}")

    net_out_src, net_out_dst = [], []
    cell_out_src, cell_out_dst = [], []
    
    # Track fanout for each net
    fanout_count = {net: 0 for net in nets}
    
    # Track gate types and their typical delays
    gate_delays = {
        'buf': 0.1,
        'and': 0.15,
        'or': 0.15,
        'nand': 0.12,
        'nor': 0.12,
        'not': 0.08,
        'xor': 0.2,
        'xnor': 0.2,
        'dff': 0.25
    }

    for inst in instances:
        if inst['instance'] not in instance_nodes:
            continue
        inst_idx = instance_nodes[inst['instance']]
        
        # Determine gate type from instance name
        gate_type = 'buf'  # default
        for gtype in gate_delays.keys():
            if gtype in inst['module'].lower():
                gate_type = gtype
                break

        for port, net in inst['connections'].items():
            if net not in net_nodes:
                continue
            net_idx = net_nodes[net]
            
            # Count fanout for each net
            fanout_count[net] += 1

            if port.upper() == 'A':
                net_out_src.append(net_idx)
                net_out_dst.append(inst_idx)
            elif port.upper() == 'X':
                cell_out_src.append(inst_idx)
                cell_out_dst.append(net_idx)
            else:
                net_out_src.append(net_idx)
                net_out_dst.append(inst_idx)
                cell_out_src.append(inst_idx)
                cell_out_dst.append(net_idx)

    data_dict = {
        ('net', 'net_out', 'cell'): (torch.tensor(net_out_src, dtype=torch.int64),
                                     torch.tensor(net_out_dst, dtype=torch.int64)),
        ('cell', 'cell_out', 'net'): (torch.tensor(cell_out_src, dtype=torch.int64),
                                     torch.tensor(cell_out_dst, dtype=torch.int64))
    }

    hg = dgl.heterograph(data_dict, num_nodes_dict={'net': num_net_nodes, 'cell': num_instance_nodes})
    g = dgl.to_homogeneous(hg)
    
    # Generate more realistic features
    num_total_nodes = g.num_nodes()
    
    # Node features based on connectivity
    node_features = []
    for i in range(num_total_nodes):
        if i < num_net_nodes:  # Net nodes
            net_name = list(net_nodes.keys())[list(net_nodes.values()).index(i)]
            fo = fanout_count[net_name]
            # Features: [is_net, is_cell, fanout, level, capacitance, ...]
            features = [1, 0, fo/max(fanout_count.values()), 0.5, 0.3] + [0]*5
        else:  # Cell nodes
            inst_name = list(instance_nodes.keys())[list(instance_nodes.values()).index(i-num_net_nodes)]
            gate_type = 'buf'
            for gtype in gate_delays.keys():
                if gtype in instances[i-num_net_nodes]['module'].lower():
                    gate_type = gtype
                    break
            # Features: [is_net, is_cell, gate_delay, drive_strength, ...]
            features = [0, 1, gate_delays[gate_type], 0.8, 0.4] + [0]*5
        node_features.append(features)
    
    g.ndata['nf'] = torch.tensor(node_features, dtype=torch.float32)
    
    # Generate timing-related features
    base_delay = torch.tensor([gate_delays['buf']], dtype=torch.float32)
    variation = torch.randn(num_total_nodes, 1) * 0.1  # 10% variation
    g.ndata['n_net_delays'] = torch.abs(base_delay + variation)
    
    # Generate arrival times based on topological levels
    g.ndata['n_ats'] = torch.zeros(num_total_nodes, 1)
    for i in range(num_total_nodes):
        in_edges = g.in_edges(i)[0]
        if len(in_edges) > 0:
            prev_delays = g.ndata['n_net_delays'][in_edges]
            g.ndata['n_ats'][i] = torch.max(prev_delays) + g.ndata['n_net_delays'][i]
    
    # Generate slews based on gate types and fanout
    g.ndata['n_slews'] = torch.abs(g.ndata['n_net_delays'] * (1 + variation))
    
    # Timing endpoints
    g.ndata['n_is_timing_endpt'] = torch.zeros(num_total_nodes, 1)
    out_degrees = g.out_degrees()
    g.ndata['n_is_timing_endpt'][out_degrees == 0] = 1
    
    # Edge features
    num_edges = g.num_edges()
    edge_features = torch.zeros(num_edges, 512)
    for i in range(num_edges):
        src, dst = g.edges()[0][i], g.edges()[1][i]
        # Basic edge features based on source and destination nodes
        edge_features[i, 0] = g.ndata['n_net_delays'][src]
        edge_features[i, 1] = g.ndata['n_net_delays'][dst]
        # Add more meaningful edge features (first 10 elements)
        edge_features[i, 2:10] = torch.tensor([
            g.ndata['nf'][src][2],  # fanout/gate_delay
            g.ndata['nf'][dst][2],  # fanout/gate_delay
            g.ndata['n_ats'][src],  # arrival time at source
            g.ndata['n_slews'][src],  # slew at source
            g.in_degrees(src),  # in-degree of source
            g.out_degrees(src),  # out-degree of source
            g.in_degrees(dst),  # in-degree of destination
            g.out_degrees(dst),  # out-degree of destination
        ])
    
    g.edata['ef'] = edge_features
    g.edata['e_cell_delays'] = torch.abs(torch.randn(num_edges, 4) * 0.1 + 0.2)  # More realistic cell delays

    return g 