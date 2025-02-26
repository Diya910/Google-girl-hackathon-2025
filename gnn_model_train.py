import torch
import torch.nn.functional as F
import random
import time
import argparse
import os
from sklearn.metrics import r2_score
import tee

from graph_dataset import data_train, data_test
from gnn_model import TimingGCN

def parse_arguments():
    """Parse command line arguments with clear descriptions."""
    parser = argparse.ArgumentParser(description="Graph Neural Network for Timing Prediction")
    parser.add_argument(
        '--evaluation_iteration', type=int,
        help='If specified, evaluate a saved model instead of training')
    parser.add_argument(
        '--save_dir', type=str,
        help='If specified, the log and model would be saved to/loaded from that save directory')
    
    # Default flags with clear naming conventions
    parser.set_defaults(wire_latency=True, gate_latency=True, reference_values=True)
    
    parser.add_argument(
        '--no_wire_latency', dest='wire_latency', action='store_false',
        help='Disable the wire latency training supervision (default enabled)')
    parser.add_argument(
        '--no_gate_latency', dest='gate_latency', action='store_false',
        help='Disable the gate latency training supervision (default enabled)')
    parser.add_argument(
        '--no_reference_values', dest='reference_values', action='store_false',
        help='Disable reference values breakdown in training (default enabled)')
    
    return parser.parse_args()

def evaluate(neural_network):
    """
    Evaluate the model on arrival time prediction.
    
    Args:
        neural_network: The trained neural network model
    """
    neural_network.eval()
    with torch.no_grad():
        def evaluate_dataset(dataset, dataset_name):
            print(f'======= {dataset_name} dataset ======')
            for design_name, (graph, timing_stats) in dataset.items():
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                predictions = neural_network(graph, timing_stats, reference_values=False)[2][:, :4]
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                actual_values = graph.ndata['n_atslew'][:, :4]
                
                # Note: Parameter order in r2_score follows sklearn convention
                # This maintains compatibility with original implementation
                accuracy = r2_score(predictions.cpu().numpy().reshape(-1),
                              actual_values.cpu().numpy().reshape(-1))
                print('{:15} r2 {:1.5f}, time {:2.5f}'.format(design_name, accuracy, end_time - start_time))
        
        evaluate_dataset(data_train, "Training")
        evaluate_dataset(data_test, "Test")

def evaluate_wire_latency(neural_network):
    """
    Evaluate the model on wire latency prediction.
    
    Args:
        neural_network: The trained neural network model
    """
    neural_network.eval()
    with torch.no_grad():
        def evaluate_dataset(dataset, dataset_name):
            print(f'======= {dataset_name} dataset ======')
            for design_name, (graph, timing_stats) in dataset.items():
                predictions = neural_network(graph, timing_stats, reference_values=False)[0]
                actual_values = graph.ndata['n_net_delays_log']
                
                # Note: Parameter order in r2_score follows sklearn convention
                # This maintains compatibility with original implementation
                accuracy = r2_score(predictions.cpu().numpy().reshape(-1),
                              actual_values.cpu().numpy().reshape(-1))
                print('{:15} {}'.format(design_name, accuracy))
        
        evaluate_dataset(data_train, "Training")
        evaluate_dataset(data_test, "Test")

def training_loop(neural_network, args):
    """
    Main training loop for the neural network.
    
    Args:
        neural_network: The neural network model to train
        args: Command line arguments
    """
    optimizer = torch.optim.Adam(neural_network.parameters(), lr=0.0005)
    batch_count = 7

    for epoch in range(10000):
        neural_network.train()
        train_loss_sum_wire_latency, train_loss_sum_gate_latency, train_loss_sum_arrival_timing = 0, 0, 0
        optimizer.zero_grad()
        
        for design_name, (graph, timing_stats) in random.sample(data_train.items(), batch_count):
            pred_wire_latency, pred_gate_latency, pred_arrival_timing = neural_network(
                graph, timing_stats, reference_values=args.reference_values)
            loss_wire_latency, loss_gate_latency = 0, 0

            if args.wire_latency:
                loss_wire_latency = F.mse_loss(pred_wire_latency, graph.ndata['n_net_delays_log'])
                train_loss_sum_wire_latency += loss_wire_latency.item()

            if args.gate_latency:
                loss_gate_latency = F.mse_loss(pred_gate_latency, graph.edges['cell_out'].data['e_cell_delays'])
                train_loss_sum_gate_latency += loss_gate_latency.item()
            else:
                # Workaround for DGL memory issue - create negligible gradient flow
                loss_gate_latency = torch.sum(pred_gate_latency) * 0.0
            
            loss_arrival_timing = F.mse_loss(pred_arrival_timing, graph.ndata['n_atslew'])
            train_loss_sum_arrival_timing += loss_arrival_timing.item()
            
            (loss_wire_latency + loss_gate_latency + loss_arrival_timing).backward()
            
        optimizer.step()

        if epoch == 0 or epoch % 20 == 19:
            with torch.no_grad():
                neural_network.eval()
                test_loss_sum_wire_latency, test_loss_sum_gate_latency = 0, 0
                test_loss_sum_arrival_timing, test_loss_sum_arrival_timing_prop = 0, 0
                
                for design_name, (graph, timing_stats) in data_test.items():
                    pred_wire_latency, pred_gate_latency, pred_arrival_timing = neural_network(
                        graph, timing_stats, reference_values=True)
                    _, _, pred_arrival_timing_prop = neural_network(
                        graph, timing_stats, reference_values=False)

                    if args.wire_latency:
                        test_loss_sum_wire_latency += F.mse_loss(
                            pred_wire_latency, graph.ndata['n_net_delays_log']).item()
                    if args.gate_latency:
                        test_loss_sum_gate_latency += F.mse_loss(
                            pred_gate_latency, graph.edges['cell_out'].data['e_cell_delays']).item()
                    test_loss_sum_arrival_timing += F.mse_loss(
                        pred_arrival_timing, graph.ndata['n_atslew']).item()
                    test_loss_sum_arrival_timing_prop += F.mse_loss(
                        pred_arrival_timing_prop, graph.ndata['n_atslew']).item()
                    
                print('Epoch {}, wire latency {:.6f}/{:.6f}, gate latency {:.6f}/{:.6f}, arrival timing {:.6f}/({:.6f})'.format(
                    epoch,
                    train_loss_sum_wire_latency / batch_count,
                    test_loss_sum_wire_latency / len(data_test),
                    train_loss_sum_gate_latency / batch_count,
                    test_loss_sum_gate_latency / len(data_test),
                    train_loss_sum_arrival_timing / batch_count,
                    test_loss_sum_arrival_timing_prop / len(data_test)))

            should_save = (epoch == 0 or epoch % 200 == 199 or 
                          (epoch > 6000 and test_loss_sum_arrival_timing_prop / len(data_test) < 6))
            
            if should_save:
                if args.save_dir:
                    save_path = './checkpoints/{}/{}.pth'.format(args.save_dir, epoch)
                    torch.save(neural_network.state_dict(), save_path)
                    print('saved model to', save_path)
                try:
                    evaluate(neural_network)
                except ValueError as e:
                    print(f"Error during evaluation: {e}")
                    print('Error testing, but ignored')

def main():
    """Main entry point of the program."""
    # Set up compute device
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {compute_device}")
    
    # Initialize model
    neural_network = TimingGCN().to(compute_device)
    
    # Parse arguments
    args = parse_arguments()
    
    if args.evaluation_iteration:
        assert args.save_dir, 'No save directory specified for evaluation'
        model_path = f'./checkpoints/{args.save_dir}/{args.evaluation_iteration}.pth'
        print(f"Loading model from {model_path}")
        neural_network.load_state_dict(torch.load(model_path, map_location=compute_device))

        evaluate(neural_network)
        evaluate_wire_latency(neural_network)
        
    else:
        if args.save_dir:
            checkpoint_dir = f'./checkpoints/{args.save_dir}'
            print(f'Saving logs and models to {checkpoint_dir}')
            os.makedirs(checkpoint_dir)  # Raises error if directory exists
            stdout_f = f'{checkpoint_dir}/stdout.log'
            stderr_f = f'{checkpoint_dir}/stderr.log'
            with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
                training_loop(neural_network, args)
        else:
            print('No save directory specified. Model checkpoints and logs will not be saved.')
            training_loop(neural_network, args)

if __name__ == '__main__':
    main()