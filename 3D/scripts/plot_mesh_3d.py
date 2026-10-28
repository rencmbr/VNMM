import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


def plot_mesh_3d(
    nodes_coords, 
    nodes_vectors, 
    output_path=None, 
    show=True, 
    caminho_saida=None, 
    exibir=None,
    vector_scale=0.15
):
    """
    Plots and visualizes 3D node locations and unit director vectors.
    
    Parameters:
    - nodes_coords: Numpy array (N, 3) of node coordinates (x, y, z).
    - nodes_vectors: Numpy array (N, 3) of vector components (tx, ty, tz).
    - output_path: Filepath to save the plot figure.
    - show: Whether to display figure window (default: True).
    - vector_scale: Scaling length for director quiver arrows.
    """
    if caminho_saida is not None:
        output_path = caminho_saida
    if exibir is not None:
        show = exibir

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    
    x = nodes_coords[:, 0]
    y = nodes_coords[:, 1]
    z = nodes_coords[:, 2]
    tx = nodes_vectors[:, 0]
    ty = nodes_vectors[:, 1]
    tz = nodes_vectors[:, 2]
    
    # Plot nodes
    ax.scatter(x, y, z, color="blue", s=40, zorder=3, label="Nodes (N=12)")
    
    # Annotate node indices
    for idx, (xi, yi, zi) in enumerate(zip(x, y, z)):
        ax.text(xi + 0.02, yi + 0.02, zi + 0.02, f"{idx}", fontsize=8, fontweight="bold", color="black")
    
    # Plot direction vectors (quiver in 3D)
    ax.quiver(
        x, y, z, 
        tx, ty, tz, 
        length=vector_scale, 
        normalize=True, 
        color="red", 
        linewidth=1.5, 
        label="Director Vectors (t)", 
        arrow_length_ratio=0.3
    )
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Nodal Mesh and Director Vector Visualization (VNMM 3D)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    
    fig.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"3D Mesh image successfully saved to: {output_path}")
        
    if show:
        plt.show()
        
    plt.close(fig)


# Backward compatibility aliases
plot_mesh = plot_mesh_3d
plotar_malha = plot_mesh_3d
plotar_malha_3d = plot_mesh_3d

