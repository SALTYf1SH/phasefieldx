"""
Visualization Example
====================

This script demonstrates how to use the cross-platform visualization module
in phasefieldx to visualize simulation results.

The visualization module supports multiple backends:
- Plotly (interactive 3D plots)
- PyVista (VTK-based 3D plots, if available)
- Matplotlib (2D plots, limited 3D functionality)

This allows users to choose the most appropriate visualization tool for their platform
and needs, without being limited to OpenGL-dependent solutions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Import the visualization module
from phasefieldx.PostProcessing.visualization import Visualizer, plot_vtu

# Define a function to demonstrate different visualization backends
def demonstrate_visualization(vtu_file_path):
    """
    Demonstrate different visualization backends for the same VTU file.

    Parameters
    ----------
    vtu_file_path : str
        Path to the VTU file to visualize.
    """
    print(f"Visualizing VTU file: {vtu_file_path}")

    # Check if the file exists
    if not os.path.exists(vtu_file_path):
        print(f"Error: File {vtu_file_path} does not exist.")
        return

    # 1. Use the convenience function with auto backend selection
    print("\n1. Using plot_vtu with auto backend selection:")
    try:
        plot_vtu(vtu_file_path, scalars='u', title="Auto Backend")
        print("  Success!")
    except Exception as e:
        print(f"  Error: {e}")

    # 2. Use Plotly backend explicitly
    print("\n2. Using Plotly backend explicitly:")
    try:
        visualizer = Visualizer(backend='plotly')
        visualizer.plot_vtu(vtu_file_path, scalars='u', title="Plotly Backend")
        print("  Success!")
    except Exception as e:
        print(f"  Error: {e}")

    # 3. Use PyVista backend explicitly
    print("\n3. Using PyVista backend explicitly:")
    try:
        visualizer = Visualizer(backend='pyvista')
        visualizer.plot_vtu(vtu_file_path, scalars='u', title="PyVista Backend")
        print("  Success!")
    except Exception as e:
        print(f"  Error: {e}")

    # 4. Use Matplotlib backend explicitly
    print("\n4. Using Matplotlib backend explicitly:")
    try:
        visualizer = Visualizer(backend='matplotlib')
        visualizer.plot_vtu(vtu_file_path, scalars='u', title="Matplotlib Backend")
        print("  Success!")
    except Exception as e:
        print(f"  Error: {e}")


# Main execution
if __name__ == "__main__":
    # Try to find a VTU file to visualize
    # First, check if we're running from an example directory
    example_dirs = [
        "1100_Traction_displacement_control",
        "examples/Elasticity/1100_Traction_displacement_control",
        "../1100_Traction_displacement_control"
    ]

    vtu_file = None
    for dir_path in example_dirs:
        if os.path.exists(dir_path):
            paraview_dir = os.path.join(dir_path, "paraview-solutions_vtu")
            if os.path.exists(paraview_dir):
                vtu_files = [f for f in os.listdir(paraview_dir) if f.endswith(".vtu")]
                if vtu_files:
                    vtu_file = os.path.join(paraview_dir, vtu_files[0])
                    break

    if vtu_file:
        print(f"Found VTU file: {vtu_file}")
        demonstrate_visualization(vtu_file)
    else:
        print("No VTU file found. Please run a simulation first or provide a path to a VTU file.")
        print("Example usage:")
        print("  python visualization_example.py /path/to/your/file.vtu")

        # If no VTU file is found, create a simple example mesh
        print("\nCreating a simple example mesh for visualization...")

        try:
            import pyvista as pv

            # Create a more complex mesh to better demonstrate surface rendering
            # Create a cube with a hole
            mesh = pv.Cube().triangulate()

            # Create a sphere to subtract from the cube
            sphere = pv.Sphere(radius=0.4, center=(0, 0, 0))

            # Boolean subtraction to create a cube with a hole
            mesh = mesh.boolean_difference(sphere)

            # Add a scalar field for coloring
            mesh["example_scalar"] = np.sin(mesh.points[:, 0] * 10) + np.cos(mesh.points[:, 1] * 10)

            # Save the mesh to a temporary file
            temp_file = "example_mesh.vtu"
            mesh.save(temp_file)

            print(f"Created example mesh: {temp_file}")
            demonstrate_visualization(temp_file)

            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)

        except ImportError:
            print("PyVista is not available to create an example mesh.")
            print("Please install PyVista or provide a path to a VTU file.")
