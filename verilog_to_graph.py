import argparse
import torch
import dgl
from pyverilog.vparser.parser import parse

def parse_verilog(file_path):
    ast, directives = parse([file_path])
    description = ast.description
    modules = [item for item in description.definitions if hasattr(item, 'name')]
    if not modules:
        raise ValueError("No module found in the Verilog file.")
    top_module = modules[0]
    return top_module

def extract_nets_and_instances(module):
    nets = set()
    instances = []

    if hasattr(module, 'portlist') and module.portlist is not None:
        for port in module.portlist.ports:
            if hasattr(port, "first") and port.first is not None:
                nets.add(port.first.name)
            else:
                nets.add(port.name)

    for item in module.items:
        if item.__class__.__name__ == 'Decl':
            for subitem in item.list:
                if subitem.__class__.__name__ == 'Wire':
                    nets.add(subitem.name)
    
    for item in module.items:
        if item.__class__.__name__ == 'InstanceList':
            for inst in item.instances:
                conn_dict = {}
                for portarg in inst.portlist:
                    port_name = portarg.portname
                    if hasattr(portarg.argname, 'name'):
                        signal = portarg.argname.name
                    else:
                        signal = str(portarg.argname)
                    conn_dict[port_name] = signal
                    nets.add(signal)
                instances.append({
                    'module': item.module,
                    'instance': inst.name,
                    'connections': conn_dict
                })
    return list(nets), instances

def build_graph(nets, instances):
    net_nodes = {net: idx for idx, net in enumerate(nets)}
    instance_nodes = {inst['instance']: idx for idx, inst in enumerate(instances)}  # Fix here

    num_net_nodes = len(nets)
    num_instance_nodes = len(instances)
    
    print(f"Total nets: {num_net_nodes}, Total instances: {num_instance_nodes}")

    net_out_src, net_out_dst = [], []
    cell_out_src, cell_out_dst = [], []

    for inst in instances:
        if inst['instance'] not in instance_nodes:
            continue
        inst_idx = instance_nodes[inst['instance']]

        for port, net in inst['connections'].items():
            if net not in net_nodes:
                continue
            net_idx = net_nodes[net]

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

    print(f"Max net ID: {max(net_out_src + net_out_dst, default=0)}")
    print(f"Max instance ID: {max(cell_out_src + cell_out_dst, default=0)}")

    data_dict = {
        ('net', 'net_out', 'cell'): (torch.tensor(net_out_src, dtype=torch.int64),
                                     torch.tensor(net_out_dst, dtype=torch.int64)),
        ('cell', 'cell_out', 'net'): (torch.tensor(cell_out_src, dtype=torch.int64),
                                     torch.tensor(cell_out_dst, dtype=torch.int64))
    }

    hg = dgl.heterograph(data_dict, num_nodes_dict={'net': num_net_nodes, 'cell': num_instance_nodes})

    g = dgl.to_homogeneous(hg)

    num_total_nodes = g.num_nodes()
    g.ndata['nf'] = torch.randn(num_total_nodes, 10)
    g.ndata['n_net_delays'] = torch.abs(torch.randn(num_total_nodes, 1))
    g.ndata['n_ats'] = torch.randn(num_total_nodes, 1)
    g.ndata['n_slews'] = torch.abs(torch.randn(num_total_nodes, 1))
    g.ndata['n_is_timing_endpt'] = torch.randint(0, 2, (num_total_nodes, 1)).float()

    num_edges = g.num_edges()
    g.edata['ef'] = torch.randn(num_edges, 512)
    g.edata['e_cell_delays'] = torch.randn(num_edges, 4)

    return g

def main():
    parser = argparse.ArgumentParser(
        description="Convert a Verilog RTL file to a DGL graph binary file for timing analysis")
    parser.add_argument('--verilog', type=str, required=True, help="Path to the Verilog RTL file (e.g., test.v)")
    parser.add_argument('--output', type=str, default="test.graph.bin", help="Output graph binary filename")
    args = parser.parse_args()
    
    top_module = parse_verilog(args.verilog)
    nets, instances = extract_nets_and_instances(top_module)
    print("Extracted nets:", nets)
    print("Extracted instances:", instances)
    
    g = build_graph(nets, instances)
    dgl.save_graphs(args.output, [g])
    print(f"Graph saved to {args.output}")

if __name__ == '__main__':
    main()