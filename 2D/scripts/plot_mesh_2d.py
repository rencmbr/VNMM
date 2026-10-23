import matplotlib.pyplot as plt


def plot_mesh(nodes_coords, nodes_vectors, output_path=None, show=True, caminho_saida=None, exibir=None):
    """
    Plots and visualizes node locations and unit director vectors.
    
    Parameters:
    - nodes_coords: Numpy array (N, 2) of node coordinates (x, y).
    - nodes_vectors: Numpy array (N, 2) of vector components (tx, ty).
    - output_path: Filepath to save the plot figure.
    - show: Whether to display figure window (default: True).
    """
    if caminho_saida is not None:
        output_path = caminho_saida
    if exibir is not None:
        show = exibir

    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = nodes_coords[:, 0]
    y = nodes_coords[:, 1]
    tx = nodes_vectors[:, 0]
    ty = nodes_vectors[:, 1]
    
    # Plot nodes
    ax.scatter(x, y, color="blue", s=30, zorder=3, label="Nodes")
    
    # Annotate node indices
    for idx, (xi, yi) in enumerate(zip(x, y)):
        ax.annotate(
            f"{idx}", 
            (xi, yi), 
            textcoords="offset points", 
            xytext=(2, 2), 
            fontsize=6, 
            fontweight="bold",
            color="black"
        )
    
    # Plot direction vectors
    ax.quiver(
        x, y, tx, ty, 
        angles="xy", 
        scale_units="xy", 
        scale=1.0, 
        color="red", 
        width=0.004, 
        label="Director Vectors (t)", 
        zorder=4
    )
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Nodal Mesh and Director Vector Visualization")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    
    fig.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Mesh image successfully saved to: {output_path}")
        
    if show:
        plt.show()
        
    plt.close(fig)


# Backward compatibility alias
plotar_malha = plot_mesh
