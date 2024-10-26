import torch
import argparse
import os
import matplotlib.pyplot as plt

def load_attention_scores(file_path):
    """
    Load the attention scores from a .pt file.
    Returns a list of attention tensors.
    Each tensor corresponds to a layer and has shape (batch_size, num_heads, seq_len, seq_len).
    """
    attention_scores = torch.load(file_path)
    return attention_scores

def compute_cosine_similarity(tensor1, tensor2):
    """
    Compute the cosine similarity between two tensors.
    The tensors are expected to be of the same shape.
    Flatten the tensors before computing cosine similarity.
    """
    # Ensure tensors are on CPU and of float type
    tensor1 = tensor1.contiguous().view(-1).float()
    tensor2 = tensor2.contiguous().view(-1).float()
    # Compute cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=0)
    return cos_sim.item()

def compare_attention_scores(attn_scores1, attn_scores2):
    """
    Compare the attention scores between two runs.
    Returns a nested list of cosine similarities for each layer, batch, and head.
    Also computes average similarities across heads, layers, and overall.
    """
    num_layers1 = len(attn_scores1)
    num_layers2 = len(attn_scores2)
    num_layers = min(num_layers1, num_layers2)
    similarities = []
    avg_similarities_per_layer = []  # Average similarity per layer (across batches and heads)
    avg_similarities_per_batch = []  # Average similarity per batch (across layers and heads)
    total_similarities = []          # Collect all similarities to compute overall average

    for layer_idx in range(num_layers):
        print("processing layer ", layer_idx)
        layer_attn1 = attn_scores1[layer_idx]  # Shape: (batch_size1, num_heads1, seq_len1, seq_len1)
        layer_attn2 = attn_scores2[layer_idx]  # Shape: (batch_size2, num_heads2, seq_len2, seq_len2)

        batch_size1, num_heads1, seq_len1, _ = layer_attn1.shape
        batch_size2, num_heads2, seq_len2, _ = layer_attn2.shape

        if batch_size1 != batch_size2:
            print(f'Warning: Batch sizes differ at layer {layer_idx+1}. Truncating to the minimum.')
        batch_size = min(batch_size1, batch_size2)

        if num_heads1 != num_heads2:
            print(f'Warning: Number of heads differ at layer {layer_idx+1}. Truncating to the minimum.')
        num_heads = min(num_heads1, num_heads2)

        seq_len = min(seq_len1, seq_len2)

        layer_sims = []
        layer_total = 0.0  # Sum of similarities in this layer
        layer_count = 0    # Number of similarities computed in this layer

        for batch_idx in range(batch_size):
            batch_sims = []
            batch_total = 0.0  # Sum of similarities in this batch
            batch_count = 0    # Number of similarities computed in this batch

            for head_idx in range(num_heads):
                head_attn1 = layer_attn1[batch_idx, head_idx, :seq_len, :seq_len]
                head_attn2 = layer_attn2[batch_idx, head_idx, :seq_len, :seq_len]
                cos_sim = compute_cosine_similarity(head_attn1, head_attn2)
                batch_sims.append(cos_sim)

                # Update totals
                batch_total += cos_sim
                layer_total += cos_sim
                total_similarities.append(cos_sim)
                batch_count += 1
                layer_count += 1

            layer_sims.append(batch_sims)

            # Compute average similarity for this batch in this layer
            avg_batch_sim = batch_total / batch_count if batch_count > 0 else 0.0
            # Append to avg_similarities_per_batch (one entry per batch)
            if len(avg_similarities_per_batch) <= batch_idx:
                avg_similarities_per_batch.append([])
            avg_similarities_per_batch[batch_idx].append(avg_batch_sim)

        similarities.append(layer_sims)

        # Compute average similarity for this layer across all batches and heads
        avg_layer_sim = layer_total / layer_count if layer_count > 0 else 0.0
        avg_similarities_per_layer.append(avg_layer_sim)

    # Compute overall average similarity
    overall_avg_similarity = sum(total_similarities) / len(total_similarities) if len(total_similarities) > 0 else 0.0

    return similarities, avg_similarities_per_layer, avg_similarities_per_batch, overall_avg_similarity

def plot_similarities(avg_sim_per_layer, overall_avg_sim, output_dir, output):
    """
    Plot the average cosine similarity per layer and overall average similarity.
    Saves the figure to the specified output directory.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    layers = list(range(1, len(avg_sim_per_layer) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(layers, avg_sim_per_layer, marker='o', label='Per-Layer Average Similarity')
    plt.hlines(overall_avg_sim, xmin=1, xmax=len(layers), colors='r', linestyles='dashed', label='Overall Average Similarity')

    plt.title('Cosine Similarity between Attention Scores per Layer')
    plt.xlabel('Layer')
    plt.ylabel('Average Cosine Similarity')
    plt.xticks(layers)

    # Adjust y-axis limits to be more compact based on data range
    min_sim = min(avg_sim_per_layer)
    max_sim = max(avg_sim_per_layer)
    y_margin = (max_sim - min_sim) * 0.1  # Add 10% margin on top and bottom
    y_min = max(0.0, min_sim - y_margin)
    y_max = min(1.0, max_sim + y_margin)
    if y_min == y_max:
        y_min = max(0.0, y_min - 0.1)
        y_max = min(1.0, y_max + 0.1)
    plt.ylim(y_min, y_max)

    plt.grid(True)
    plt.legend()

    # Save the plot
    plot_path = os.path.join(os.path.join(output_dir, f'cosine_similarity_per_layer_{output}.png'))
    plt.savefig(plot_path)
    plt.close()
    print(f'Plot saved to {plot_path}')

def main():
    parser = argparse.ArgumentParser(description='Compare attention scores between two runs.')
    parser.add_argument('--file1', type=str, required=True, help='Path to the first attention scores file.')
    parser.add_argument('--file2', type=str, required=True, help='Path to the second attention scores file.')
    parser.add_argument('--output', type=str, default='cosine_similarities.txt', help='Output file to save the similarities.')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the output files and plots.')
    args = parser.parse_args()

    # Load attention scores
    attn_scores1 = load_attention_scores(args.file1)
    attn_scores2 = load_attention_scores(args.file2)

    # Compare attention scores
    similarities, avg_sim_per_layer, avg_sim_per_batch, overall_avg_sim = compare_attention_scores(attn_scores1, attn_scores2)

    # Save similarities to file
    output_path = os.path.join(args.output_dir, args.output) + ".txt"
    with open(output_path, 'w') as f:
        f.write(f'Overall Average Cosine Similarity: {overall_avg_sim:.6f}\n\n')

        f.write('Average Similarity per Layer:\n')
        for layer_idx, avg_sim in enumerate(avg_sim_per_layer):
            f.write(f'Layer {layer_idx+1}: Average Cosine Similarity = {avg_sim:.6f}\n')
        f.write('\n')

        f.write('Average Similarity per Batch:\n')
        for batch_idx, batch_avg_sims in enumerate(avg_sim_per_batch):
            # Average over layers for this batch
            avg_sim = sum(batch_avg_sims) / len(batch_avg_sims) if len(batch_avg_sims) > 0 else 0.0
            f.write(f'Batch {batch_idx+1}: Average Cosine Similarity = {avg_sim:.6f}\n')
        f.write('\n')

        f.write('Detailed Cosine Similarities:\n')
        for layer_idx, layer_sims in enumerate(similarities):
            f.write(f'Layer {layer_idx+1}:\n')
            for batch_idx, batch_sims in enumerate(layer_sims):
                f.write(f'  Batch {batch_idx+1}:\n')
                for head_idx, sim in enumerate(batch_sims):
                    f.write(f'    Head {head_idx+1}: Cosine Similarity = {sim:.6f}\n')
            f.write('\n')

    print(f'Cosine similarities have been saved to {output_path}')

    # Plot similarities
    plot_similarities(avg_sim_per_layer, overall_avg_sim, args.output_dir, args.output)

if __name__ == '__main__':
    main()
