"""
Torus Unwrapping Visualizer - Enhanced visualization for Collapse Geometry

This module implements various 2D projection techniques to unwrap the toroidal
manifold into different flat representations, preserving different aspects of the
relational structure.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.patches import Circle, Wedge, Rectangle, FancyArrowPatch
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from collections import defaultdict
import math

# Import the 3D torus simulator for consistency
from axiom7.visualizer.torus_simulation_visualizer import TorusSimulationVisualizer


class TorusUnwrappingVisualizer:
    """
    A visualizer that provides various methods to unwrap the toroidal manifold
    into 2D representations, preserving different relational properties.
    Uses TorusSimulationVisualizer for consistent 3D-to-2D mapping.
    """
    
    def __init__(self, 
                 field_resolution: int = 100, 
                 adaptation_rate: float = 0.4,
                 memory_weight: float = 0.6,
                 vector_density: int = 15):
        """
        Initialize the torus unwrapping visualizer.
        
        Args:
            field_resolution: Resolution of field grid for visualization
            adaptation_rate: Rate at which field projections adapt to changes
            memory_weight: Influence of memory on field projection
            vector_density: Density of vector field visualization
        """
        # Create 3D simulator to maintain consistency
        self.simulator = TorusSimulationVisualizer(field_resolution=field_resolution)
        
        # Core parameters
        self.field_resolution = field_resolution
        self.adaptation_rate = adaptation_rate
        self.memory_weight = memory_weight
        self.vector_density = vector_density
        
        # Field dynamics parameters
        self.wave_complexity = 4  # Number of cycles in the field pattern
        self.symmetry_factor = 4  # Controls symmetry of the field pattern
        self.field_smoothing = 1.5  # Amount of smoothing applied to fields
        
        # Visualization configuration
        self.show_relations = True
        self.show_field = True
        self.show_vortices = True
        self.show_grains = True
        self.edge_alpha = 0.6
        self.node_alpha = 0.8
        self.field_alpha = 0.7
        
        # Field visualization parameters
        self.kernel_size_factor = 0.5    # Kernel size for field influence
        self.kernel_size_offset = 0.5    # Base kernel size
        
        # Create grid for visualization
        theta = np.linspace(0, 2*np.pi, self.field_resolution)
        phi = np.linspace(0, 2*np.pi, self.field_resolution)
        self._theta_grid, self._phi_grid = np.meshgrid(theta, phi)
        
        # Field data
        self._awareness_field = np.zeros((field_resolution, field_resolution))
        self._phase_field = np.zeros((field_resolution, field_resolution))
        self._tension_field = np.zeros((field_resolution, field_resolution))
        self._polarity_field_u = np.zeros((field_resolution, field_resolution))
        self._polarity_field_v = np.zeros((field_resolution, field_resolution))
        
        # Track time for field dynamics
        self._current_time = 0.0
        
        # Setup colormaps
        self._setup_colormaps()
    
    def _setup_colormaps(self):
        """Setup specialized colormaps for field visualization"""
        # Create custom colormaps for different field properties
        self.colormaps = {
            'awareness': self._create_awareness_colormap(),
            'phase': self._create_phase_colormap(),
            'tension': self._create_tension_colormap(),
            'polarity': self._create_polarity_colormap(),
            'void': self._create_void_colormap(),
            'vortex': self._create_vortex_colormap(),
            'transition': self._create_transition_colormap()
        }
    
    def _create_awareness_colormap(self):
        """Create a colormap for awareness visualization that matches the reference"""
        return mcolors.LinearSegmentedColormap.from_list(
            'awareness_cmap',
            [(0.0, '#081B41'),    # Deep dark blue (void/low awareness)
             (0.3, '#3A4FB8'),    # Medium blue (emerging awareness)
             (0.6, '#8A64D6'),    # Purple (active awareness)
             (0.8, '#2EB18C'),    # Teal/green (high awareness)
             (1.0, '#FFE74C')],   # Bright gold (peak awareness)
            N=256
        )
    
    def _create_phase_colormap(self):
        """Create a colormap for phase visualization"""
        # Use circular colormap for phase
        return plt.cm.hsv
    
    def _create_tension_colormap(self):
        """Create a colormap for field tension visualization"""
        return mcolors.LinearSegmentedColormap.from_list(
            'tension_cmap',
            [(0.0, '#FFFFFF'),    # White (no tension)
             (0.3, '#FFE8C1'),    # Light yellow (low tension)
             (0.6, '#FFC53F'),    # Gold (medium tension)
             (0.8, '#FF7D36'),    # Orange (high tension)
             (1.0, '#E3170A')],   # Red (maximum tension)
            N=256
        )
    
    def _create_polarity_colormap(self):
        """Create a colormap for polarity direction"""
        return mcolors.LinearSegmentedColormap.from_list(
            'polarity_cmap',
            [(0.0, '#9A3B67'),    # Dark magenta (strong negative)
             (0.3, '#E481AA'),    # Light magenta (weak negative)
             (0.5, '#FFFFFF'),    # White (neutral)
             (0.7, '#A5C0E1'),    # Light blue (weak positive)
             (1.0, '#3166A1')],   # Deep blue (strong positive)
            N=256
        )
    
    def _create_void_colormap(self):
        """Create a colormap for void visualization"""
        return mcolors.LinearSegmentedColormap.from_list(
            'void_cmap',
            [(0.0, '#FFFFFF00'),  # Transparent white (no void)
             (0.3, '#1F1F1F40'),  # Semi-transparent dark gray
             (0.6, '#08080880'),  # More opaque dark gray
             (1.0, '#000000C0')], # Very dark, mostly opaque
            N=256
        )
    
    def _create_vortex_colormap(self):
        """Create a colormap for vortex visualization"""
        return mcolors.LinearSegmentedColormap.from_list(
            'vortex_cmap',
            [(0.0, '#FFFFFF00'),   # Transparent (no vortex)
             (0.3, '#FF00FF40'),   # Semi-transparent magenta
             (0.6, '#FF00FF80'),   # More opaque magenta
             (1.0, '#FF00FFC0')],  # Bright magenta
            N=256
        )
    
    def _create_transition_colormap(self):
        """Create a colormap for phase transitions"""
        return mcolors.LinearSegmentedColormap.from_list(
            'transition_cmap',
            [(0.0, '#E0F7FA'),    # Light cyan (stable)
             (0.4, '#4DD0E1'),    # Cyan (low transition)
             (0.7, '#0097A7'),    # Teal (medium transition)
             (1.0, '#006064')],   # Dark teal (high transition)
            N=256
        )
    
    def update_state(self, manifold):
        """
        Update the visualizer state based on the current manifold state.
        Uses the 3D simulator for consistent state management.
        
        Args:
            manifold: The relational manifold
        """
        # Update time from manifold
        self._current_time = getattr(manifold, 'time', self._current_time + 0.1)
        
        # Update 3D simulator state first
        self.simulator.update_state(manifold)
        
        # Get grain positions directly from simulator
        self._grain_positions = self.simulator._grain_positions
        
        # Calculate 2D field representation by generating standing wave patterns and adding grain influences
        self._generate_field_patterns(manifold)
    
    def _generate_field_patterns(self, manifold):
        """
        Generate field patterns based on standing waves and grain influences.
        This is the core method that creates the visualized field dynamics.
        
        Args:
            manifold: The relational manifold
        """
        # Generate base standing wave pattern
        self._generate_standing_wave_field()
        
        # Incorporate grain influences
        self._incorporate_grain_influences(manifold)
        
        # Apply post-processing and ensure field periodicity
        self._post_process_fields()
        self._ensure_field_periodicity()
    
    def _generate_standing_wave_field(self):
        """
        Generate a standing wave pattern for the awareness field.
        This creates the characteristic wave pattern similar to the reference image.
        """
        # Parameters
        symmetry = self.symmetry_factor  # Number of waves around each axis
        time_phase = self._current_time * 0.05  # Slow time evolution
        
        # Generate wave patterns
        for i in range(self.field_resolution):
            for j in range(self.field_resolution):
                theta = self._theta_grid[i, j]
                phi = self._phi_grid[i, j]
                
                # Create wave pattern along theta axis (major circle)
                wave_theta = np.cos(symmetry * theta + time_phase)
                
                # Create wave pattern along phi axis (minor circle)
                wave_phi = np.cos(symmetry * phi + time_phase * 0.7)
                
                # Create diagonal pattern for added complexity
                diag1 = 0.4 * np.sin((symmetry-1) * (theta - phi) + time_phase * 0.5)
                diag2 = 0.4 * np.sin((symmetry-1) * (theta + phi) + time_phase * 0.3)
                
                # Combine waves with weights to create pattern
                wave_value = (wave_theta * 0.6 + 
                             wave_phi * 0.4 + 
                             diag1 * 0.3 + 
                             diag2 * 0.3)
                
                # Scale to 0-1 range
                wave_value = (wave_value + 1.5) / 3.0  
                wave_value = np.clip(wave_value, 0.0, 1.0)
                
                # Apply power curve to enhance contrast
                wave_value = wave_value ** 1.2
                
                # Set awareness field value
                self._awareness_field[i, j] = wave_value
                
                # Create complementary phase field (in range 0-2π)
                self._phase_field[i, j] = (theta + phi + time_phase * 2) % (2 * np.pi)
                
                # Calculate polarity field components (gradient of awareness)
                # These will be updated in post-processing
                self._polarity_field_u[i, j] = 0.0
                self._polarity_field_v[i, j] = 0.0
    
    def _incorporate_grain_influences(self, manifold):
        """
        Incorporate grain influences into the field.
        Grains affect the base field pattern with their awareness and relations.
        
        Args:
            manifold: The relational manifold
        """
        # Skip if no grains
        if not manifold.grains:
            return
            
        # Calculate influence radius based on resolution
        influence_radius = 2 * np.pi / (self.field_resolution * 0.25)
        
        # Apply grain influences to field
        for grain_id, grain in manifold.grains.items():
            if grain_id not in self._grain_positions:
                continue
                
            # Get grain position and properties
            grain_theta, grain_phi = self._grain_positions[grain_id]
            grain_awareness = grain.awareness
            
            # Skip grains with negligible awareness
            if grain_awareness < 0.1:
                continue
                
            # Calculate influence kernel based on awareness
            kernel_size = self.kernel_size_offset + grain_awareness * self.kernel_size_factor
            
            # Apply grain influence to nearby field points
            for i in range(self.field_resolution):
                for j in range(self.field_resolution):
                    theta = self._theta_grid[i, j]
                    phi = self._phi_grid[i, j]
                    
                    # Calculate toroidal distance (shortest path)
                    d_theta = min(abs(theta - grain_theta), 2*np.pi - abs(theta - grain_theta))
                    d_phi = min(abs(phi - grain_phi), 2*np.pi - abs(phi - grain_phi))
                    dist_sq = d_theta**2 + d_phi**2
                    
                    # Skip points too far away
                    if dist_sq > influence_radius**2:
                        continue
                        
                    # Calculate influence strength (Gaussian kernel)
                    influence = np.exp(-dist_sq / (2 * kernel_size**2))
                    
                    # Scale influence by grain awareness 
                    influence *= grain_awareness * 0.3
                    
                    # Add to awareness field (amplify existing pattern)
                    self._awareness_field[i, j] += influence * self._awareness_field[i, j]
                    
                    # Affect phase field (slight phase adjustment)
                    if hasattr(grain, 'phase_memory'):
                        grain_phase = grain.phase_memory % (2*np.pi)
                        phase_influence = influence * 0.2
                        phase_diff = ((grain_phase - self._phase_field[i, j] + np.pi) % (2*np.pi)) - np.pi
                        self._phase_field[i, j] += phase_diff * phase_influence
                        self._phase_field[i, j] = self._phase_field[i, j] % (2*np.pi)
                    
                    # Add to tension field if grain has unresolved tension
                    if hasattr(grain, 'unresolved_tension') and grain.unresolved_tension > 0:
                        tension_influence = influence * grain.unresolved_tension * 0.5
                        self._tension_field[i, j] += tension_influence
    
    def _post_process_fields(self):
        """
        Apply post-processing to the fields for better visualization.
        This includes normalization, smoothing, and contrast enhancement.
        """
        # Normalize awareness field to 0-1 range
        if np.max(self._awareness_field) > 0:
            self._awareness_field = self._awareness_field / np.max(self._awareness_field)
        
        # Apply smoothing for a more natural appearance
        self._awareness_field = gaussian_filter(self._awareness_field, sigma=self.field_smoothing)
        self._tension_field = gaussian_filter(self._tension_field, sigma=1.0)
        
        # Ensure phase field is in range 0-2π
        self._phase_field = self._phase_field % (2 * np.pi)
        
        # Normalize tension field
        if np.max(self._tension_field) > 0:
            self._tension_field = self._tension_field / np.max(self._tension_field)
        
        # Calculate polarity field (gradient of awareness)
        grad_y, grad_x = np.gradient(self._awareness_field)
        self._polarity_field_u = -grad_x  # Negative x-gradient maps to u
        self._polarity_field_v = -grad_y  # Negative y-gradient maps to v
        
        # Normalize polarity field
        polarity_magnitude = np.sqrt(self._polarity_field_u**2 + self._polarity_field_v**2)
        max_magnitude = np.max(polarity_magnitude)
        if max_magnitude > 0:
            self._polarity_field_u /= max_magnitude
            self._polarity_field_v /= max_magnitude
    
    def _ensure_field_periodicity(self):
        """
        Ensure fields are periodic at boundaries.
        This makes the visualization seamless across the torus.
        """
        # Smooth awareness field across theta boundary (left-right edges)
        for i in range(self.field_resolution):
            # Average edge values
            left_edge = self._awareness_field[i, 0:3].mean()
            right_edge = self._awareness_field[i, -3:].mean()
            avg_edge = (left_edge + right_edge) / 2
            
            # Apply smoothing to boundaries
            for j in range(3):
                weight = (3-j)/3
                self._awareness_field[i, j] = self._awareness_field[i, j] * (1-weight) + avg_edge * weight
                self._awareness_field[i, -(j+1)] = self._awareness_field[i, -(j+1)] * (1-weight) + avg_edge * weight
        
        # Smooth awareness field across phi boundary (top-bottom edges)
        for j in range(self.field_resolution):
            # Average edge values
            top_edge = self._awareness_field[0:3, j].mean()
            bottom_edge = self._awareness_field[-3:, j].mean()
            avg_edge = (top_edge + bottom_edge) / 2
            
            # Apply smoothing to boundaries
            for i in range(3):
                weight = (3-i)/3
                self._awareness_field[i, j] = self._awareness_field[i, j] * (1-weight) + avg_edge * weight
                self._awareness_field[-(i+1), j] = self._awareness_field[-(i+1), j] * (1-weight) + avg_edge * weight
        
        # Apply similar smoothing to other fields
        # Tension field
        for i in range(self.field_resolution):
            left_edge = self._tension_field[i, 0:3].mean()
            right_edge = self._tension_field[i, -3:].mean()
            avg_edge = (left_edge + right_edge) / 2
            
            for j in range(3):
                weight = (3-j)/3
                self._tension_field[i, j] = self._tension_field[i, j] * (1-weight) + avg_edge * weight
                self._tension_field[i, -(j+1)] = self._tension_field[i, -(j+1)] * (1-weight) + avg_edge * weight
        
        for j in range(self.field_resolution):
            top_edge = self._tension_field[0:3, j].mean()
            bottom_edge = self._tension_field[-3:, j].mean()
            avg_edge = (top_edge + bottom_edge) / 2
            
            for i in range(3):
                weight = (3-i)/3
                self._tension_field[i, j] = self._tension_field[i, j] * (1-weight) + avg_edge * weight
                self._tension_field[-(i+1), j] = self._tension_field[-(i+1), j] * (1-weight) + avg_edge * weight
    
    def create_standard_unwrapping(self, manifold, 
                                color_by='awareness',
                                show_relations=True,
                                show_vortices=True,
                                show_void_regions=True):
        """
        Create a standard unwrapping of the torus as a rectangle.
        This is the most common way to visualize a torus in 2D.
        
        Args:
            manifold: The relational manifold
            color_by: Field property to use for coloring ('awareness', 'phase', 'tension')
            show_relations: Whether to show relations between grains
            show_vortices: Whether to show vortices
            show_void_regions: Whether to show void regions
            
        Returns:
            Figure and axes objects
        """
        # Update state based on manifold
        self.update_state(manifold)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Determine color field and colormap
        if color_by == 'awareness':
            color_field = self._awareness_field
            colormap = self.colormaps['awareness']
            colorbar_label = 'Awareness'
        elif color_by == 'phase':
            color_field = self._phase_field / (2*np.pi)  # Normalize to 0-1
            colormap = self.colormaps['phase']
            colorbar_label = 'Phase'
        elif color_by == 'tension':
            color_field = self._tension_field
            colormap = self.colormaps['tension']
            colorbar_label = 'Field Tension'
        elif color_by == 'void':
            color_field = np.zeros_like(self._awareness_field)  # Default to zeros
            if hasattr(manifold, 'find_void_regions'):
                # Add void regions to the field
                void_regions = manifold.find_void_regions()
                for void in void_regions:
                    if 'theta' in void and 'phi' in void:
                        theta, phi = void['theta'], void['phi']
                        strength = void.get('strength', 0.5)
                        radius = void.get('radius', 0.3)
                        
                        # Add void to field
                        for i in range(self.field_resolution):
                            for j in range(self.field_resolution):
                                grid_theta = self._theta_grid[i, j]
                                grid_phi = self._phi_grid[i, j]
                                
                                # Calculate toroidal distance
                                d_theta = min(abs(grid_theta - theta), 2*np.pi - abs(grid_theta - theta))
                                d_phi = min(abs(grid_phi - phi), 2*np.pi - abs(grid_phi - phi))
                                dist_sq = d_theta**2 + d_phi**2
                                
                                # Add void influence based on distance
                                void_influence = strength * np.exp(-dist_sq / (2 * radius**2))
                                color_field[i, j] = max(color_field[i, j], void_influence)
            
            colormap = self.colormaps['void']
            colorbar_label = 'Void Presence'
        else:
            # Default to awareness
            color_field = self._awareness_field
            colormap = self.colormaps['awareness']
            colorbar_label = 'Awareness'
        
        # Plot field as background - always show field
        field = ax.pcolormesh(self._theta_grid, self._phi_grid, color_field,
                           cmap=colormap, alpha=self.field_alpha, shading='auto')
        
        # Add colorbar
        cbar = fig.colorbar(field, ax=ax)
        cbar.set_label(colorbar_label)
        
        # Show relations between grains
        if show_relations:
            for grain_id, relations in self.simulator._relation_strengths.items():
                if grain_id not in self._grain_positions or grain_id not in manifold.grains:
                    continue
                    
                theta1, phi1 = self._grain_positions[grain_id]
                
                for related_id, strength in relations.items():
                    if related_id not in self._grain_positions or related_id not in manifold.grains:
                        continue
                        
                    theta2, phi2 = self._grain_positions[related_id]
                    
                    # Check for wrap-around
                    d_theta = abs(theta1 - theta2)
                    d_phi = abs(phi1 - phi2)
                    
                    if d_theta > np.pi or d_phi > np.pi:
                        # Handle wrap-around for visualization
                        # This improves connection visualization across torus boundaries
                        
                        # Calculate new coordinates that maintain the shortest path
                        new_theta2 = theta2
                        new_phi2 = phi2
                        
                        # Check and adjust theta (horizontal wrap)
                        if d_theta > np.pi:
                            if theta1 > theta2:
                                new_theta2 += 2*np.pi  # Move right point to the right side of wrap
                            else:
                                new_theta2 -= 2*np.pi  # Move right point to the left side of wrap
                        
                        # Check and adjust phi (vertical wrap)
                        if d_phi > np.pi:
                            if phi1 > phi2:
                                new_phi2 += 2*np.pi  # Move bottom point to the top side of wrap
                            else:
                                new_phi2 -= 2*np.pi  # Move top point to the bottom side of wrap
                        
                        # Draw connection with wrap indication
                        # Draw dashed line to indicate wrap
                        ax.plot([theta1, new_theta2], [phi1, new_phi2], 
                               color='gray', linewidth=1, alpha=0.5, linestyle='--')
                        continue
                    
                    # Line width based on relation strength
                    lw = max(0.5, strength * 3)
                    
                    # Check if there's memory value
                    memory = 0.0
                    grain = manifold.grains[grain_id]
                    if hasattr(grain, 'relation_memory') and related_id in grain.relation_memory:
                        memory = grain.relation_memory[related_id]
                    
                    # Line color based on memory polarity
                    if memory > 0.1:
                        color = plt.cm.Reds(min(1.0, abs(memory)))
                    elif memory < -0.1:
                        color = plt.cm.Blues(min(1.0, abs(memory)))
                    else:
                        color = 'gray'
                    
                    # Draw relation
                    ax.plot([theta1, theta2], [phi1, phi2], color=color, linewidth=lw, alpha=self.edge_alpha)
        
        # Show grain nodes
        if self.show_grains:
            for grain_id, grain in manifold.grains.items():
                if grain_id not in self._grain_positions:
                    continue
                    
                # Get grain position
                theta, phi = self._grain_positions[grain_id]
                
                # Node size based on awareness
                size = 100 * grain.awareness
                
                # Node color
                if color_by == 'awareness':
                    color = colormap(grain.awareness)
                elif color_by == 'phase':
                    if hasattr(manifold, 'get_phase_continuity'):
                        phase = manifold.get_phase_continuity(grain_id) % (2*np.pi) / (2*np.pi)
                    else:
                        phase = 0.5
                    color = colormap(phase)
                elif color_by == 'tension':
                    if hasattr(grain, 'field_tension'):
                        tension = grain.field_tension
                    elif hasattr(grain, 'unresolved_tension'):
                        tension = grain.unresolved_tension
                    else:
                        tension = 0.0
                    color = colormap(tension)
                elif color_by == 'void':
                    void_presence = manifold.get_void_presence(grain_id) if hasattr(manifold, 'get_void_presence') else 0.0
                    color = colormap(void_presence)
                else:
                    color = 'blue'
                
                # Plot grain node
                ax.scatter(theta, phi, s=size, color=color, alpha=self.node_alpha, edgecolors='black', linewidths=0.5)
        
        # Show vortices if requested
        if show_vortices and hasattr(manifold, 'detect_vortices'):
            vortices = manifold.detect_vortices()
            
            for vortex in vortices:
                # Get vortex location
                if 'center_node' in vortex:
                    center_id = vortex['center_node']
                    if center_id in self._grain_positions:
                        theta, phi = self._grain_positions[center_id]
                    else:
                        continue
                elif 'theta' in vortex and 'phi' in vortex:
                    theta, phi = vortex['theta'], vortex['phi']
                else:
                    continue
                
                # Vortex strength and direction
                strength = vortex.get('strength', 0.5)
                direction = vortex.get('rotation_direction', 'clockwise')
                
                # Size based on strength
                size = 200 * strength
                
                # Color based on direction
                color = 'red' if direction == 'clockwise' else 'blue'
                
                # Draw vortex
                ax.scatter(theta, phi, s=size, color=color, alpha=0.7, 
                          marker='o', edgecolors='white', linewidths=1.5)
                
                # Add label for strong vortices
                if strength > 0.7:
                    ax.text(theta, phi, f"V:{strength:.2f}", color='white', fontsize=8)
        
        # Show void regions if requested
        if show_void_regions and hasattr(manifold, 'find_void_regions'):
            void_regions = manifold.find_void_regions()
            
            for void in void_regions:
                # Get void center
                if 'center_point' in void:
                    center_id = void['center_point']
                    if center_id in self._grain_positions:
                        theta, phi = self._grain_positions[center_id]
                    else:
                        continue
                elif 'theta' in void and 'phi' in void:
                    theta, phi = void['theta'], void['phi']
                else:
                    continue
                
                # Get void properties
                strength = void.get('strength', 0.5)
                radius = void.get('radius', 0.3)
                
                # Draw void as dark circle
                void_size = 150 * radius * strength
                
                ax.scatter(theta, phi, s=void_size, color='black', alpha=0.7, 
                          marker='o', edgecolors='darkgray', linewidths=1)
                
                # Add label for strong voids
                if strength > 0.7:
                    ax.text(theta, phi, f"Void:{strength:.2f}", color='white', fontsize=8)
        
        # Add hints for torus topology (showing which edges are connected)
        # Left edge connects to right edge
        arrow_count = 5
        for i in range(arrow_count):
            phi = (i + 0.5) * 2 * np.pi / arrow_count
            # Arrow from left to right
            ax.annotate('', xy=(2*np.pi, phi), xytext=(0, phi),
                      arrowprops=dict(arrowstyle='<->', color='gray', alpha=0.5))
        
        # Bottom edge connects to top edge
        for i in range(arrow_count):
            theta = (i + 0.5) * 2 * np.pi / arrow_count
            # Arrow from bottom to top
            ax.annotate('', xy=(theta, 2*np.pi), xytext=(theta, 0),
                      arrowprops=dict(arrowstyle='<->', color='gray', alpha=0.5))
        
        # Set axis labels and title
        ax.set_xlabel('θ (Major Circle - Around Center)')
        ax.set_ylabel('φ (Minor Circle - Around Tube)')
        
        # Set axis limits
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(0, 2*np.pi)
        
        # Set tick marks at π intervals
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add title
        plt.title(f'Standard Torus Unwrapping (t={manifold.time:.1f})')
        
        return fig, ax
    
    def create_vector_field_visualization(self, manifold, base_field='awareness'):
        """
        Create a visualization of the vector field (polarity) on the torus.
        
        Args:
            manifold: The relational manifold
            base_field: Base field to visualize ('awareness', 'phase', 'tension')
            
        Returns:
            Figure and axes objects
        """
        # Update state
        self.update_state(manifold)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get base field and colormap
        if base_field == 'awareness':
            color_field = self._awareness_field
            colormap = self.colormaps['awareness']
            colorbar_label = 'Awareness'
        elif base_field == 'phase':
            color_field = self._phase_field / (2*np.pi)  # Normalize to 0-1
            colormap = self.colormaps['phase']
            colorbar_label = 'Phase'
        elif base_field == 'tension':
            color_field = self._tension_field
            colormap = self.colormaps['tension']
            colorbar_label = 'Field Tension'
        else:
            color_field = self._awareness_field
            colormap = self.colormaps['awareness']
            colorbar_label = 'Awareness'
        
        # Plot base field
        base = ax.pcolormesh(self._theta_grid, self._phi_grid, color_field,
                          cmap=colormap, alpha=0.7, shading='auto')
        
        # Add colorbar
        cbar = fig.colorbar(base, ax=ax)
        cbar.set_label(colorbar_label)
        
        # Create downsampled grid for vector field visualization
        skip = self.field_resolution // self.vector_density
        theta_points = np.linspace(0, 2*np.pi, self.vector_density)
        phi_points = np.linspace(0, 2*np.pi, self.vector_density)
        
        # Get polarity field components
        u = self._polarity_field_u[::skip, ::skip]
        v = self._polarity_field_v[::skip, ::skip]
        
        # Draw vector field
        ax.quiver(theta_points, phi_points, u, v, 
                pivot='mid', scale=30, width=0.003, 
                color='white', alpha=0.7)
        
        # Show grain nodes
        if self.show_grains:
            for grain_id, grain in manifold.grains.items():
                if grain_id not in self._grain_positions:
                    continue
                    
                # Get grain position
                theta, phi = self._grain_positions[grain_id]
                
                # Node size based on awareness
                size = 80 * grain.awareness
                
                # Plot grain node
                ax.scatter(theta, phi, s=size, color='white', alpha=0.7, 
                          edgecolors='black', linewidths=0.5)
        
        # Add hints for torus topology (showing which edges are connected)
        # Left edge connects to right edge
        arrow_count = 5
        for i in range(arrow_count):
            phi = (i + 0.5) * 2 * np.pi / arrow_count
            # Arrow from left to right
            ax.annotate('', xy=(2*np.pi, phi), xytext=(0, phi),
                      arrowprops=dict(arrowstyle='<->', color='gray', alpha=0.5))
        
        # Bottom edge connects to top edge
        for i in range(arrow_count):
            theta = (i + 0.5) * 2 * np.pi / arrow_count
            # Arrow from bottom to top
            ax.annotate('', xy=(theta, 2*np.pi), xytext=(theta, 0),
                      arrowprops=dict(arrowstyle='<->', color='gray', alpha=0.5))
        
        # Set axis labels and title
        ax.set_xlabel('θ (Major Circle - Around Center)')
        ax.set_ylabel('φ (Minor Circle - Around Tube)')
        
        # Set axis limits
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(0, 2*np.pi)
        
        # Set tick marks at π intervals
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add title
        plt.title(f'Vector Field Visualization (t={manifold.time:.1f})')
        
        return fig, ax
    
    def create_phase_domain_visualization(self, manifold):
        """
        Create a visualization of phase domains on the unwrapped torus.
        
        Args:
            manifold: The relational manifold
            
        Returns:
            Figure and axes objects
        """
        # Update state
        self.update_state(manifold)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Draw phase field as background
        phase_field = self._phase_field / (2*np.pi)  # Normalize to 0-1
        ax.pcolormesh(self._theta_grid, self._phi_grid, phase_field,
                    cmap=self.colormaps['phase'], alpha=0.4, shading='auto')
        
        # Check if manifold supports domain identification
        if hasattr(manifold, 'find_field_emergent_patterns'):
            patterns = manifold.find_field_emergent_patterns()
            
            if isinstance(patterns, dict) and 'phase_domains' in patterns:
                # Set up domain colors
                domain_colors = {
                    'solid': 'darkblue',
                    'liquid': 'royalblue',
                    'gas': 'lightgreen',
                    'radiant': 'gold',
                    'frozen': 'lightgray',
                    'viscous': 'purple',
                    'default': 'orange'
                }
                
                # Draw domain regions
                for domain in patterns['phase_domains']:
                    domain_type = domain.get('phase_type', 'default')
                    domain_points = domain.get('points', [])
                    
                    # Skip empty domains
                    if not domain_points:
                        continue
                    
                    # Get domain color
                    color = domain_colors.get(domain_type, domain_colors['default'])
                    
                    # Get domain center
                    if 'theta_center' in domain and 'phi_center' in domain:
                        center_theta = domain['theta_center']
                        center_phi = domain['phi_center']
                    else:
                        # Calculate average position
                        thetas = []
                        phis = []
                        for grain_id in domain_points:
                            if grain_id in self._grain_positions:
                                theta, phi = self._grain_positions[grain_id]
                                thetas.append(theta)
                                phis.append(phi)
                        
                        if thetas and phis:
                            # Calculate circular mean for correct wrapping
                            sin_sum_theta = sum(np.sin(t) for t in thetas)
                            cos_sum_theta = sum(np.cos(t) for t in thetas)
                            center_theta = np.arctan2(sin_sum_theta, cos_sum_theta) % (2*np.pi)
                            
                            sin_sum_phi = sum(np.sin(p) for p in phis)
                            cos_sum_phi = sum(np.cos(p) for p in phis)
                            center_phi = np.arctan2(sin_sum_phi, cos_sum_phi) % (2*np.pi)
                        else:
                            # Skip domain if we can't determine center
                            continue
                    
                    # Draw domain center marker
                    ax.scatter(center_theta, center_phi, s=200, color=color,
                              alpha=0.7, edgecolors='white', linewidths=1.5)
                    
                    # Add domain label
                    ax.text(center_theta, center_phi, domain_type,
                           color='white', fontsize=9, ha='center', va='center')
                    
                    # Draw domain points
                    for grain_id in domain_points:
                        if grain_id in self._grain_positions:
                            theta, phi = self._grain_positions[grain_id]
                            ax.scatter(theta, phi, s=50, color=color, alpha=0.5)
                
                # Add legend
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                              markersize=10, label=domain_type)
                    for domain_type, color in domain_colors.items()
                    if domain_type != 'default'
                ]
                
                ax.legend(handles=legend_elements, loc='upper right')
        
        # Draw all grain nodes
        for grain_id, grain in manifold.grains.items():
            if grain_id not in self._grain_positions:
                continue
                
            # Get grain position
            theta, phi = self._grain_positions[grain_id]
            
            # Node size based on awareness
            size = 30 * grain.awareness
            
            # Node color based on phase
            if hasattr(manifold, 'get_phase_continuity'):
                phase = manifold.get_phase_continuity(grain_id) % (2*np.pi) / (2*np.pi)
            elif hasattr(grain, 'phase_memory'):
                phase = grain.phase_memory % (2*np.pi) / (2*np.pi)
            else:
                phase = 0.5
            
            # Plot grain node
            ax.scatter(theta, phi, s=size, color=self.colormaps['phase'](phase), 
                      alpha=0.7, edgecolors='black', linewidths=0.5)
        
        # Add hints for torus topology (showing which edges are connected)
        # Left edge connects to right edge
        arrow_count = 5
        for i in range(arrow_count):
            phi = (i + 0.5) * 2 * np.pi / arrow_count
            # Arrow from left to right
            ax.annotate('', xy=(2*np.pi, phi), xytext=(0, phi),
                      arrowprops=dict(arrowstyle='<->', color='gray', alpha=0.5))
        
        # Bottom edge connects to top edge
        for i in range(arrow_count):
            theta = (i + 0.5) * 2 * np.pi / arrow_count
            # Arrow from bottom to top
            ax.annotate('', xy=(theta, 2*np.pi), xytext=(theta, 0),
                      arrowprops=dict(arrowstyle='<->', color='gray', alpha=0.5))
        
        # Set axis labels and title
        ax.set_xlabel('θ (Major Circle - Around Center)')
        ax.set_ylabel('φ (Minor Circle - Around Tube)')
        
        # Set axis limits
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(0, 2*np.pi)
        
        # Set tick marks at π intervals
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add title
        plt.title(f'Phase Domain Visualization (t={manifold.time:.1f})')
        
        return fig, ax