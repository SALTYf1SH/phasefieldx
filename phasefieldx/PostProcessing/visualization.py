"""
Visualization Module
===================

This module provides cross-platform visualization capabilities for phasefieldx.
It supports multiple backends for visualization, including:
- Matplotlib (2D plots)
- Plotly (interactive 3D plots)
- PyVista (VTK-based 3D plots, if available)

This allows users to choose the most appropriate visualization tool for their platform
and needs, without being limited to OpenGL-dependent solutions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Try to import optional dependencies
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    warnings.warn("PyVista not available. Some 3D visualization features will be limited.")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Install with 'pip install plotly' for interactive 3D visualization.")


class Visualizer:
    """
    A class to handle visualization of simulation results with multiple backends.

    This class provides methods to visualize VTU files and other simulation data
    using different visualization libraries, allowing for cross-platform compatibility.
    """

    def __init__(self, backend='auto'):
        """
        Initialize the Visualizer with the specified backend.

        Parameters
        ----------
        backend : str, optional
            The visualization backend to use. Options are:
            - 'auto': Automatically select the best available backend
            - 'plotly': Use Plotly for visualization
            - 'pyvista': Use PyVista for visualization
            - 'matplotlib': Use Matplotlib for visualization
            Default is 'auto'.
        """
        self.backend = backend

        # Determine the best available backend if 'auto' is selected
        if self.backend == 'auto':
            if PLOTLY_AVAILABLE:
                self.backend = 'plotly'
            elif PYVISTA_AVAILABLE:
                self.backend = 'pyvista'
            else:
                self.backend = 'matplotlib'
                warnings.warn("Using matplotlib as fallback. 3D visualization will be limited.")

    def plot_vtu(self, file_path, scalars=None, cpos='xy', show_scalar_bar=True,
                 show_edges=False, title=None, save_path=None, **kwargs):
        """
        Plot data from a VTU file using the selected backend.

        Parameters
        ----------
        file_path : str
            Path to the VTU file to visualize.
        scalars : str, optional
            Name of the scalar field to visualize.
        cpos : str or list, optional
            Camera position. For PyVista, this can be 'xy', 'xz', 'yz', or a list of positions.
            For Plotly, this affects the initial view.
        show_scalar_bar : bool, optional
            Whether to show the scalar bar (colorbar).
        show_edges : bool, optional
            Whether to show mesh edges.
        title : str, optional
            Title for the plot.
        save_path : str, optional
            Path to save the visualization. If None, the plot is displayed.
        **kwargs : dict
            Additional keyword arguments to pass to the backend's plotting function.

        Returns
        -------
        fig : object
            The figure object created by the backend.
        """
        if self.backend == 'pyvista':
            return self._plot_vtu_pyvista(file_path, scalars, cpos, show_scalar_bar,
                                         show_edges, title, save_path, **kwargs)
        elif self.backend == 'plotly':
            return self._plot_vtu_plotly(file_path, scalars, cpos, show_scalar_bar,
                                        show_edges, title, save_path, **kwargs)
        else:
            return self._plot_vtu_matplotlib(file_path, scalars, title, save_path, **kwargs)

    def _plot_vtu_pyvista(self, file_path, scalars=None, cpos='xy', show_scalar_bar=True,
                          show_edges=False, title=None, save_path=None, **kwargs):
        """
        Plot VTU file using PyVista.
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is not available. Install with 'pip install pyvista'.")

        try:
            # Try to use Xvfb if available (for headless environments)
            try:
                pv.start_xvfb()
            except Exception:
                pass

            # Read the VTU file
            mesh = pv.read(file_path)

            # Create a plotter
            plotter = pv.Plotter()

            # Add the mesh to the plotter
            plotter.add_mesh(mesh, scalars=scalars, show_edges=show_edges,
                            show_scalar_bar=show_scalar_bar, **kwargs)

            # Set the camera position
            plotter.camera_position = cpos

            # Set the title
            if title:
                plotter.add_title(title)

            # Save or show the plot
            if save_path:
                plotter.screenshot(save_path)
                return plotter
            else:
                plotter.show()
                return plotter

        except Exception as e:
            warnings.warn(f"PyVista visualization failed: {e}. Falling back to alternative backend.")
            if PLOTLY_AVAILABLE:
                return self._plot_vtu_plotly(file_path, scalars, cpos, show_scalar_bar,
                                           show_edges, title, save_path, **kwargs)
            else:
                return self._plot_vtu_matplotlib(file_path, scalars, title, save_path, **kwargs)

    def _plot_vtu_plotly(self, file_path, scalars=None, cpos='xy', show_scalar_bar=True,
                         show_edges=False, title=None, save_path=None, **kwargs):
        """
        Plot VTU file using Plotly.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is not available. Install with 'pip install plotly'.")

        # We need PyVista to read the VTU file
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required to read VTU files. Install with 'pip install pyvista'.")

        # Read the VTU file
        mesh = pv.read(file_path)

        # Create a 3D figure
        fig = go.Figure()

        # Check if we have a 2D or 3D mesh
        is_3d = mesh.points.shape[1] > 2

        # Get the scalar data for coloring
        if scalars and scalars in mesh.array_names:
            scalar_data = mesh[scalars]

            # Normalize scalar data for coloring
            if len(scalar_data.shape) > 1 and scalar_data.shape[1] > 1:
                # Vector data - use magnitude
                scalar_data = np.linalg.norm(scalar_data, axis=1)

            # Create a colormap
            colorscale = 'Viridis'
        else:
            scalar_data = None
            colorscale = None

        # Extract mesh faces for surface plotting
        try:
            # Try to extract surface from the mesh
            if hasattr(mesh, 'extract_surface'):
                surface = mesh.extract_surface()
            else:
                surface = mesh

            # Get faces as triangles
            faces = []
            if hasattr(surface, 'faces'):
                # PyVista mesh format: [n, id1, id2, ..., idn, m, id1, id2, ...]
                i = 0
                while i < len(surface.faces):
                    n_points = surface.faces[i]
                    if n_points == 3:  # Triangle
                        faces.append([surface.faces[i+1], surface.faces[i+2], surface.faces[i+3]])
                    elif n_points == 4:  # Quad - split into two triangles
                        faces.append([surface.faces[i+1], surface.faces[i+2], surface.faces[i+3]])
                        faces.append([surface.faces[i+1], surface.faces[i+3], surface.faces[i+4]])
                    i += n_points + 1
            elif hasattr(surface, 'cells'):
                # Extract triangular faces from cells
                if hasattr(surface, 'cell_connectivity'):
                    conn = surface.cell_connectivity
                    offset = surface.offset
                    for i in range(len(offset)-1):
                        start, end = offset[i], offset[i+1]
                        if end - start == 3:  # Triangle
                            faces.append([conn[start], conn[start+1], conn[start+2]])
                        elif end - start == 4:  # Quad - split into two triangles
                            faces.append([conn[start], conn[start+1], conn[start+2]])
                            faces.append([conn[start], conn[start+2], conn[start+3]])

            # If we have faces, create a mesh3d plot
            if faces:
                # Convert faces to the format required by Plotly
                i, j, k = [], [], []
                for face in faces:
                    if len(face) >= 3:  # Ensure we have at least 3 vertices
                        i.append(face[0])
                        j.append(face[1])
                        k.append(face[2])

                # Get vertex coordinates
                x = surface.points[:, 0]
                y = surface.points[:, 1]
                z = surface.points[:, 2] if is_3d else np.zeros(len(surface.points))

                # Create the mesh3d trace
                mesh_trace = go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i, j=j, k=k,
                    intensity=scalar_data if scalar_data is not None else None,
                    colorscale=colorscale,
                    opacity=0.9,
                    showscale=show_scalar_bar,
                    colorbar=dict(title=scalars) if scalars else None,
                    name=scalars if scalars else 'Mesh'
                )

                fig.add_trace(mesh_trace)

                # Add edges if requested
                if show_edges:
                    # Create edge traces
                    edge_sets = set()
                    for face in faces:
                        for i in range(len(face)):
                            edge = tuple(sorted([face[i], face[(i+1) % len(face)]]))
                            edge_sets.add(edge)

                    for edge in edge_sets:
                        fig.add_trace(go.Scatter3d(
                            x=[x[edge[0]], x[edge[1]]],
                            y=[y[edge[0]], y[edge[1]]],
                            z=[z[edge[0]], z[edge[1]]],
                            mode='lines',
                            line=dict(color='black', width=1),
                            showlegend=False
                        ))
            else:
                # Fallback to scatter plot if we couldn't extract faces
                self._add_scatter_plot(fig, surface.points, scalar_data, scalars, show_scalar_bar, is_3d)

        except Exception as e:
            warnings.warn(f"Failed to create mesh plot: {e}. Falling back to scatter plot.")
            # Fallback to scatter plot
            self._add_scatter_plot(fig, mesh.points, scalar_data, scalars, show_scalar_bar, is_3d)

        # Set the layout
        fig.update_layout(
            title=title,
            scene=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )

        # Set initial camera position based on cpos
        if cpos == 'xy':
            fig.update_layout(scene_camera=dict(eye=dict(x=0, y=0, z=2)))
        elif cpos == 'xz':
            fig.update_layout(scene_camera=dict(eye=dict(x=0, y=2, z=0)))
        elif cpos == 'yz':
            fig.update_layout(scene_camera=dict(eye=dict(x=2, y=0, z=0)))

        # Save or show the plot
        if save_path:
            fig.write_image(save_path)
        else:
            fig.show()

        return fig

    def _add_scatter_plot(self, fig, points, scalar_data, scalars, show_scalar_bar, is_3d):
        """
        Add a scatter plot to the figure as a fallback visualization.
        """
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2] if is_3d else np.zeros(len(points)),
            mode='markers',
            marker=dict(
                size=3,
                color=scalar_data if scalar_data is not None else 'blue',
                colorscale='Viridis' if scalar_data is not None else None,
                showscale=show_scalar_bar if scalar_data is not None else False,
                colorbar=dict(title=scalars) if scalars else None,
            ),
            name=scalars if scalars else 'Mesh'
        ))

    def _plot_vtu_matplotlib(self, file_path, scalars=None, title=None, save_path=None, **kwargs):
        """
        Plot VTU file using Matplotlib (limited functionality).
        """
        # We need PyVista to read the VTU file
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required to read VTU files. Install with 'pip install pyvista'.")

        # Read the VTU file
        mesh = pv.read(file_path)

        # Extract points for plotting
        points = mesh.points

        # Create a figure
        fig, ax = plt.subplots()

        # Plot the points as a scatter plot
        if scalars and scalars in mesh.array_names:
            scalar_data = mesh[scalars]

            # Normalize scalar data for coloring
            if len(scalar_data.shape) > 1 and scalar_data.shape[1] > 1:
                # Vector data - use magnitude
                scalar_data = np.linalg.norm(scalar_data, axis=1)

            scatter = ax.scatter(points[:, 0], points[:, 1], c=scalar_data, cmap='viridis')
            plt.colorbar(scatter, ax=ax, label=scalars)
        else:
            ax.scatter(points[:, 0], points[:, 1], color='blue')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if title:
            ax.set_title(title)

        # Set aspect ratio
        ax.set_aspect('equal')

        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        return fig


# Convenience function to visualize VTU files
def plot_vtu(file_path, backend='auto', **kwargs):
    """
    Convenience function to visualize a VTU file.

    Parameters
    ----------
    file_path : str
        Path to the VTU file to visualize.
    backend : str, optional
        The visualization backend to use. Options are:
        - 'auto': Automatically select the best available backend
        - 'plotly': Use Plotly for visualization
        - 'pyvista': Use PyVista for visualization
        - 'matplotlib': Use Matplotlib for visualization
        Default is 'auto'.
    **kwargs : dict
        Additional keyword arguments to pass to the Visualizer.plot_vtu method.

    Returns
    -------
    fig : object
        The figure object created by the backend.
    """
    visualizer = Visualizer(backend=backend)
    return visualizer.plot_vtu(file_path, **kwargs)
