"""
Torus Simulation Visualizer - Enhanced visualization for Collapse Geometry

This module implements a 3D visualization approach that renders the relational
manifold as a true torus in 3D space, preserving topological relationships.
It focuses on showing the actual torus structure with field dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from collections import defaultdict
import math

# Import base visualizer class
from axiom7.visualizer.visualizer import FieldVisualizer


class TorusSimulationVisualizer(FieldVisualizer):
    """
    A 3D torus visualizer that renders the relational manifold as a true torus
    in 3D space, preserving topological relationships and showing field dynamics.
    """
    
    def __init__(self, 
             major_radius: float = 3.0, 
             minor_radius: float = 1.0,
             resolution: int = 36,
             field_resolution: int = None,  # Add this parameter
             adaptation_rate: float = 0.4,
             memory_weight: float = 0.6):
        """
        Initialize the torus simulation visualizer.
    
        Args:
            major_radius: Major radius of the torus (distance from center to middle of tube)
            minor_radius: Minor radius of the torus (radius of the tube)
            resolution: Number of points to use for torus rendering
            field_resolution: Alternative parameter name for resolution (for compatibility)
            adaptation_rate: Rate at which torus positions adapt to changes
            memory_weight: Influence of memory on torus positions
        """
        super().__init__()
    
        # Use field_resolution if provided, otherwise use resolution
        self.resolution = field_resolution if field_resolution is not None else resolution
    
        # Core torus parameters
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.adaptation_rate = adaptation_rate
        self.memory_weight = memory_weight
        
        # Field dynamics parameters
        self.wave_complexity = 4  # Number of cycles in the field pattern (higher = more waves)
        self.field_smoothing = 1.5  # Smoothing factor for field visualization
        self.symmetry_factor = 4  # Controls symmetry of the field pattern
        self.pattern_scale = 0.8  # Scale factor for pattern intensity
        
        # Relational mapping state - for torus positioning
        self._grain_positions = {}     # Maps grain_id -> (theta, phi) positions on torus
        self._memory_embeddings = {}   # Maps grain_id -> memory embedding (for continuity)
        self._relation_strengths = defaultdict(dict)  # Maps grain_id -> {related_id: strength}
        self._ancestry_couplings = defaultdict(set)  # Maps grain_id -> set of coupled ancestors
        
        # Visualization configuration
        self.show_relations = True
        self.show_field = True
        self.show_vortices = True
        self.show_grains = True
        self.edge_alpha = 0.6
        self.node_alpha = 0.8
        self.field_alpha = 0.5
        
        # Field representation for visualization
        self._theta_grid = None
        self._phi_grid = None
        self._torus_x = None
        self._torus_y = None
        self._torus_z = None
        self._awareness_field = None
        self._phase_field = None
        self._tension_field = None
        
        # 3D torus surface and colormaps
        self._setup_torus_surface()
        self._setup_colormaps()
        
        # Track time for field dynamics
        self._current_time = 0.0
    
    def _setup_torus_surface(self):
        """Initialize the torus surface grid for 3D rendering"""
        # Create grid for torus surface
        theta = np.linspace(0, 2*np.pi, self.resolution)
        phi = np.linspace(0, 2*np.pi, self.resolution)
        self._theta_grid, self._phi_grid = np.meshgrid(theta, phi)
        
        # Calculate 3D coordinates of torus surface (initial, will be updated with fields)
        # Standard parametric equations for torus:
        # x = (R + r * cos(phi)) * cos(theta)
        # y = (R + r * cos(phi)) * sin(theta)
        # z = r * sin(phi)
        
        R = self.major_radius
        r = self.minor_radius
        
        self._torus_x = (R + r * np.cos(self._phi_grid)) * np.cos(self._theta_grid)
        self._torus_y = (R + r * np.cos(self._phi_grid)) * np.sin(self._theta_grid)
        self._torus_z = r * np.sin(self._phi_grid)
        
        # Initialize field values
        self._awareness_field = np.zeros_like(self._theta_grid)
        self._phase_field = np.zeros_like(self._theta_grid)
        self._tension_field = np.zeros_like(self._theta_grid)
    
    def _setup_colormaps(self):
        """Setup specialized colormaps for field visualization"""
        # Create custom colormaps for different field properties
        self.colormaps = {
            'awareness': self._create_awareness_colormap(),
            'phase': self._create_phase_colormap(),
            'tension': self._create_tension_colormap(),
            'polarity': self._create_polarity_colormap(),
            'void': self._create_void_colormap(),
            'vortex': self._create_vortex_colormap()
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
        # Circular colormap for phase
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
    
    def update_state(self, manifold):
        """
        Update the visualizer state based on the current manifold state.
        Integrates with toroidal referencing in the core files.
        
        Args:
            manifold: The relational manifold
        """
        # Update time from manifold
        self._current_time = getattr(manifold, 'time', self._current_time + 0.1)
        
        # Get current grains
        current_grains = set(manifold.grains.keys())
        
        # Update relation strengths from manifold
        for grain_id, grain in manifold.grains.items():
            for related_id, strength in grain.relations.items():
                if related_id in manifold.grains:
                    self._relation_strengths[grain_id][related_id] = strength
        
        # Update ancestry couplings
        for grain_id, grain in manifold.grains.items():
            ancestors = getattr(grain, 'ancestry', set())
            self._ancestry_couplings[grain_id] = ancestors.intersection(current_grains)
        
        # Update memory embeddings for continuity
        for grain_id, grain in manifold.grains.items():
            if grain_id not in self._memory_embeddings:
                self._memory_embeddings[grain_id] = np.random.random(4)  # 4D embedding
            
            # Update embedding based on relation memory for continuity
            if hasattr(grain, 'relation_memory'):
                memory_update = np.zeros(4)
                memory_count = 0
                
                for related_id, memory_value in grain.relation_memory.items():
                    if related_id in self._memory_embeddings and related_id in current_grains:
                        memory_factor = np.clip(memory_value, -1.0, 1.0)
                        memory_update += self._memory_embeddings[related_id] * memory_factor
                        memory_count += 1
                
                if memory_count > 0:
                    memory_update /= memory_count
                    self._memory_embeddings[grain_id] = (1 - self.memory_weight) * self._memory_embeddings[grain_id] + \
                                                     self.memory_weight * memory_update
        
        # === CRITICAL: Integrate with toroidal referencing in core files ===
        # Get grain positions from manifold's toroidal phase tracking
        for grain_id in current_grains:
            if hasattr(manifold, 'get_toroidal_phase'):
                # Get position directly from manifold's toroidal referencing
                theta, phi = manifold.get_toroidal_phase(grain_id)
                self._grain_positions[grain_id] = (theta, phi)
            elif grain_id not in self._grain_positions:
                # Initialize random position if not available
                self._grain_positions[grain_id] = (np.random.uniform(0, 2*np.pi), 
                                              np.random.uniform(0, 2*np.pi))
        
        # Apply relational dynamics to update positions if needed
        if not hasattr(manifold, 'get_toroidal_phase'):
            self._update_grain_positions(manifold)
        
        # Calculate field values on torus with enhanced dynamics
        self._calculate_torus_fields(manifold)
    
    def _update_grain_positions(self, manifold):
        """
        Update grain positions based on relational dynamics.
        Only used as fallback if manifold doesn't have toroidal referencing.
        
        Args:
            manifold: The relational manifold
        """
        # Get current grains
        current_grains = set(manifold.grains.keys())
        
        # Collect position updates
        position_updates = {grain_id: [0.0, 0.0] for grain_id in current_grains}
        
        # Apply memory-based updates
        for grain_id in current_grains:
            if grain_id not in self._memory_embeddings:
                continue
            
            memory_vec = self._memory_embeddings[grain_id]
            
            # Project memory to toroidal coordinates
            memory_theta = np.arctan2(memory_vec[1], memory_vec[0]) % (2*np.pi)
            memory_phi = np.arctan2(memory_vec[3], memory_vec[2]) % (2*np.pi)
            
            # Current position
            current_theta, current_phi = self._grain_positions[grain_id]
            
            # Calculate differences (on circle)
            d_theta = np.sin(memory_theta - current_theta)
            d_phi = np.sin(memory_phi - current_phi)
            
            # Update based on memory influence
            position_updates[grain_id][0] += d_theta * 0.2
            position_updates[grain_id][1] += d_phi * 0.2
        
        # Apply relation-based updates
        for grain_id in current_grains:
            grain = manifold.grains[grain_id]
            theta1, phi1 = self._grain_positions[grain_id]
            
            # Process each relation
            for related_id, strength in grain.relations.items():
                if related_id not in current_grains:
                    continue
                
                theta2, phi2 = self._grain_positions[related_id]
                
                # Calculate shortest-path differences on torus
                d_theta = np.sin(theta2 - theta1)
                d_phi = np.sin(phi2 - phi1)
                
                # Apply attraction based on relation strength
                attraction = strength * 0.1
                position_updates[grain_id][0] += d_theta * attraction
                position_updates[grain_id][1] += d_phi * attraction
        
        # Apply ancestry coupling
        for grain_id in current_grains:
            ancestors = self._ancestry_couplings.get(grain_id, set())
            
            if ancestors:
                # Get current position
                theta1, phi1 = self._grain_positions[grain_id]
                
                # Calculate average ancestor position (using circular mean)
                x_sum_theta = 0.0
                y_sum_theta = 0.0
                x_sum_phi = 0.0
                y_sum_phi = 0.0
                ancestor_count = 0
                
                for ancestor_id in ancestors:
                    if ancestor_id in self._grain_positions:
                        a_theta, a_phi = self._grain_positions[ancestor_id]
                        
                        # Accumulate coordinates on unit circle for circular mean
                        x_sum_theta += np.cos(a_theta)
                        y_sum_theta += np.sin(a_theta)
                        x_sum_phi += np.cos(a_phi)
                        y_sum_phi += np.sin(a_phi)
                        ancestor_count += 1
                
                if ancestor_count > 0:
                    # Calculate circular means
                    avg_theta = np.arctan2(y_sum_theta, x_sum_theta) % (2*np.pi)
                    avg_phi = np.arctan2(y_sum_phi, x_sum_phi) % (2*np.pi)
                    
                    # Calculate differences
                    d_theta = np.sin(avg_theta - theta1)
                    d_phi = np.sin(avg_phi - phi1)
                    
                    # Apply ancestry coupling
                    ancestry_strength = 0.15 * min(1.0, ancestor_count / 3)
                    position_updates[grain_id][0] += d_theta * ancestry_strength
                    position_updates[grain_id][1] += d_phi * ancestry_strength
        
        # Apply updates with toroidal wrapping
        for grain_id in current_grains:
            current_theta, current_phi = self._grain_positions[grain_id]
            
            # Update with adaptation rate
            new_theta = (current_theta + position_updates[grain_id][0] * self.adaptation_rate) % (2*np.pi)
            new_phi = (current_phi + position_updates[grain_id][1] * self.adaptation_rate) % (2*np.pi)
            
            self._grain_positions[grain_id] = (new_theta, new_phi)
    
    def _calculate_torus_fields(self, manifold):
        """
        Calculate field values on the torus surface based on enhanced field dynamics
        that match the reference implementation.
        
        Args:
            manifold: The relational manifold
        """
        # Reset fields
        self._awareness_field = np.zeros_like(self._theta_grid)
        self._phase_field = np.zeros_like(self._theta_grid)
        self._tension_field = np.zeros_like(self._theta_grid)
        
        # First, generate base field patterns using standing waves
        self._generate_standing_wave_field()
        
        # Then incorporate grain influences
        self._incorporate_grain_influences(manifold)
        
        # Apply field normalization and smoothing
        self._post_process_fields()
        
        # Ensure field periodicity at boundaries
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
        for i in range(self.resolution):
            for j in range(self.resolution):
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
                self._awareness_field[i, j] = wave_value * self.pattern_scale
                
                # Create complementary phase field (in range 0-2π)
                self._phase_field[i, j] = (theta + phi + time_phase * 2) % (2 * np.pi)
                
                # Calculate tension field (highest in transition regions)
                # Tension is proportional to the gradient magnitude
                if i > 0 and i < self.resolution-1 and j > 0 and j < self.resolution-1:
                    # Simple gradient approximation
                    grad_theta = (self._awareness_field[i, j+1] - self._awareness_field[i, j-1]) / 2
                    grad_phi = (self._awareness_field[i+1, j] - self._awareness_field[i-1, j]) / 2
                    gradient_mag = np.sqrt(grad_theta**2 + grad_phi**2)
                    
                    # Tension is highest at medium awareness levels with high gradient
                    self._tension_field[i, j] = gradient_mag * (4 * wave_value * (1 - wave_value))
    
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
        influence_radius = 2 * np.pi / (self.resolution * 0.25)
        
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
            kernel_size = 0.2 + grain_awareness * 0.3
            
            # Apply grain influence to nearby field points
            for i in range(self.resolution):
                for j in range(self.resolution):
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
    
    def _ensure_field_periodicity(self):
        """
        Ensure fields are periodic at boundaries.
        This makes the visualization seamless across the torus.
        """
        # Smooth awareness field across theta boundary (left-right edges)
        for i in range(self.resolution):
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
        for j in range(self.resolution):
            # Average edge values
            top_edge = self._awareness_field[0:3, j].mean()
            bottom_edge = self._awareness_field[-3:, j].mean()
            avg_edge = (top_edge + bottom_edge) / 2
            
            # Apply smoothing to boundaries
            for i in range(3):
                weight = (3-i)/3
                self._awareness_field[i, j] = self._awareness_field[i, j] * (1-weight) + avg_edge * weight
                self._awareness_field[-(i+1), j] = self._awareness_field[-(i+1), j] * (1-weight) + avg_edge * weight
        
        # Apply similar smoothing to tension field
        # Theta boundary (left-right)
        for i in range(self.resolution):
            left_edge = self._tension_field[i, 0:3].mean()
            right_edge = self._tension_field[i, -3:].mean()
            avg_edge = (left_edge + right_edge) / 2
            
            for j in range(3):
                weight = (3-j)/3
                self._tension_field[i, j] = self._tension_field[i, j] * (1-weight) + avg_edge * weight
                self._tension_field[i, -(j+1)] = self._tension_field[i, -(j+1)] * (1-weight) + avg_edge * weight
        
        # Phi boundary (top-bottom)
        for j in range(self.resolution):
            top_edge = self._tension_field[0:3, j].mean()
            bottom_edge = self._tension_field[-3:, j].mean()
            avg_edge = (top_edge + bottom_edge) / 2
            
            for i in range(3):
                weight = (3-i)/3
                self._tension_field[i, j] = self._tension_field[i, j] * (1-weight) + avg_edge * weight
                self._tension_field[-(i+1), j] = self._tension_field[-(i+1), j] * (1-weight) + avg_edge * weight
    
    def _calculate_3d_position(self, theta, phi):
        """
        Calculate 3D position on torus from toroidal coordinates.
        
        Args:
            theta: Angle around major circle (0 to 2π)
            phi: Angle around minor circle (0 to 2π)
            
        Returns:
            Tuple of (x, y, z) coordinates in 3D space
        """
        R = self.major_radius
        r = self.minor_radius
        
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        
        return (x, y, z)
    
    def render_3d_torus(self, manifold, 
                       color_by='awareness',
                       show_relations=True,
                       show_vortices=True,
                       view_angle=(30, 45),
                       show_void_regions=True):
        """
        Render the manifold as a 3D torus with field properties.
        
        Args:
            manifold: The relational manifold
            color_by: Field property to use for coloring ('awareness', 'phase', 'tension')
            show_relations: Whether to show relations between grains
            show_vortices: Whether to show vortices
            view_angle: (elevation, azimuth) viewing angle
            show_void_regions: Whether to show void regions
            
        Returns:
            Figure and axes objects
        """
        # Update state based on manifold
        self.update_state(manifold)
        
        # Create 3D figure
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
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
        else:
            # Default to awareness
            color_field = self._awareness_field
            colormap = self.colormaps['awareness']
            colorbar_label = 'Awareness'
        
        # Plot torus surface with field-based coloring
        torus_surface = ax.plot_surface(
            self._torus_x, self._torus_y, self._torus_z,
            facecolors=colormap(color_field),
            alpha=self.field_alpha,
            antialiased=True,
            shade=True
        )
        
        # Add colorbar
        m = cm.ScalarMappable(cmap=colormap)
        m.set_array(color_field)
        cbar = fig.colorbar(m, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label(colorbar_label)
        
        # Show grain nodes on torus
        if self.show_grains:
            for grain_id, grain in manifold.grains.items():
                if grain_id not in self._grain_positions:
                    continue
                    
                # Get grain position
                theta, phi = self._grain_positions[grain_id]
                x, y, z = self._calculate_3d_position(theta, phi)
                
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
                else:
                    color = 'blue'
                
                # Plot grain node
                ax.scatter([x], [y], [z], s=size, color=color, alpha=self.node_alpha, edgecolors='black', linewidths=0.5)
        
        # Show relations between grains
        if show_relations:
            for grain_id, relations in self._relation_strengths.items():
                if grain_id not in self._grain_positions or grain_id not in manifold.grains:
                    continue
                    
                theta1, phi1 = self._grain_positions[grain_id]
                x1, y1, z1 = self._calculate_3d_position(theta1, phi1)
                
                for related_id, strength in relations.items():
                    if related_id not in self._grain_positions or related_id not in manifold.grains:
                        continue
                        
                    theta2, phi2 = self._grain_positions[related_id]
                    x2, y2, z2 = self._calculate_3d_position(theta2, phi2)
                    
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
                    ax.plot([x1, x2], [y1, y2], [z1, z2], color=color, linewidth=lw, alpha=self.edge_alpha)
        
        # Show vortices if requested
        if show_vortices and hasattr(manifold, 'detect_vortices'):
            vortices = manifold.detect_vortices()
            
            for vortex in vortices:
                # Get vortex location
                if 'center_node' in vortex:
                    center_id = vortex['center_node']
                    if center_id in self._grain_positions:
                        theta, phi = self._grain_positions[center_id]
                elif 'theta' in vortex and 'phi' in vortex:
                    theta, phi = vortex['theta'], vortex['phi']
                else:
                    continue
                
                # Calculate 3D position
                x, y, z = self._calculate_3d_position(theta, phi)
                
                # Vortex strength and direction
                strength = vortex.get('strength', 0.5)
                direction = vortex.get('rotation_direction', 'clockwise')
                
                # Size based on strength
                size = 200 * strength
                
                # Color based on direction
                color = 'red' if direction == 'clockwise' else 'blue'
                
                # Draw vortex
                ax.scatter([x], [y], [z], s=size, color=color, alpha=0.7, 
                          marker='o', edgecolors='white', linewidths=1.5)
                
                # Add label for strong vortices
                if strength > 0.7:
                    ax.text(x, y, z, f"V:{strength:.2f}", color='white', fontsize=8)
        
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
                
                # Draw void as dark sphere
                x, y, z = self._calculate_3d_position(theta, phi)
                void_size = 150 * radius * strength
                
                ax.scatter([x], [y], [z], s=void_size, color='black', alpha=0.7, 
                          marker='o', edgecolors='darkgray', linewidths=1)
                
                # Add label for strong voids
                if strength > 0.7:
                    ax.text(x, y, z, f"Void:{strength:.2f}", color='white', fontsize=8)
        
        # Set view angle
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Add title
        plt.title(f'3D Torus Visualization (t={manifold.time:.1f})')
        
        return fig, ax
    
    def render_field_cross_section(self, manifold,
                                  slice_type='theta',
                                  slice_value=0.0,
                                  field_property='awareness'):
        """
        Render a cross-section of the field on the torus.
        
        Args:
            manifold: The relational manifold
            slice_type: Type of slice ('theta' or 'phi')
            slice_value: Value to slice at (0 to 2π)
            field_property: Field property to visualize
            
        Returns:
            Figure and axes objects
        """
        # Update state
        self.update_state(manifold)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize slice value to 0-2π
        slice_value = slice_value % (2*np.pi)
        
        # Determine which field to use
        if field_property == 'awareness':
            field = self._awareness_field
            colormap = self.colormaps['awareness']
            title_property = 'Awareness'
        elif field_property == 'phase':
            field = self._phase_field / (2*np.pi)  # Normalize to 0-1
            colormap = self.colormaps['phase']
            title_property = 'Phase'
        elif field_property == 'tension':
            field = self._tension_field
            colormap = self.colormaps['tension']
            title_property = 'Field Tension'
        else:
            field = self._awareness_field
            colormap = self.colormaps['awareness']
            title_property = 'Awareness'
        
        # Create arrays for cross-section
        if slice_type == 'theta':
            # Find the closest theta value in the grid
            theta_index = np.argmin(np.abs(np.linspace(0, 2*np.pi, self.resolution) - slice_value))
            
            # Extract values along this theta
            x_values = np.linspace(0, 2*np.pi, self.resolution)  # phi values
            field_values = field[:, theta_index]
            
            # Plot cross-section
            ax.plot(x_values, field_values, 'k-', linewidth=2, alpha=0.7)
            ax.fill_between(x_values, 0, field_values, color=colormap(0.7), alpha=0.5)
            
            # Set labels
            ax.set_xlabel('φ (Minor Circle Angle)')
            ax.set_ylabel(title_property)
            plt.title(f'{title_property} Cross-Section at θ={slice_value:.2f} (t={manifold.time:.1f})')
            
            # Add grains on the cross-section
            for grain_id, grain in manifold.grains.items():
                if grain_id not in self._grain_positions:
                    continue
                    
                grain_theta, grain_phi = self._grain_positions[grain_id]
                
                # Check if grain is close to the theta slice
                theta_diff = min(abs(grain_theta - slice_value), 2*np.pi - abs(grain_theta - slice_value))
                
                if theta_diff < 0.2:  # Within tolerance
                    # Get grain property value
                    if field_property == 'awareness':
                        prop_value = grain.awareness
                    elif field_property == 'phase':
                        if hasattr(manifold, 'get_phase_continuity'):
                            prop_value = manifold.get_phase_continuity(grain_id) % (2*np.pi) / (2*np.pi)
                        else:
                            prop_value = 0.5
                    elif field_property == 'tension':
                        if hasattr(grain, 'field_tension'):
                            prop_value = grain.field_tension
                        elif hasattr(grain, 'unresolved_tension'):
                            prop_value = grain.unresolved_tension
                        else:
                            prop_value = 0.0
                    else:
                        prop_value = 0.5
                    
                    # Plot grain as point
                    ax.scatter(grain_phi, prop_value, s=100*grain.awareness, 
                              color=colormap(prop_value), edgecolors='black', alpha=0.8)
        
        elif slice_type == 'phi':
            # Find the closest phi value in the grid
            phi_index = np.argmin(np.abs(np.linspace(0, 2*np.pi, self.resolution) - slice_value))
            
            # Extract values along this phi
            x_values = np.linspace(0, 2*np.pi, self.resolution)  # theta values
            field_values = field[phi_index, :]
            
            # Plot cross-section
            ax.plot(x_values, field_values, 'k-', linewidth=2, alpha=0.7)
            ax.fill_between(x_values, 0, field_values, color=colormap(0.7), alpha=0.5)
            
            # Set labels
            ax.set_xlabel('θ (Major Circle Angle)')
            ax.set_ylabel(title_property)
            plt.title(f'{title_property} Cross-Section at φ={slice_value:.2f} (t={manifold.time:.1f})')
            
            # Add grains on the cross-section
            for grain_id, grain in manifold.grains.items():
                if grain_id not in self._grain_positions:
                    continue
                    
                grain_theta, grain_phi = self._grain_positions[grain_id]
                
                # Check if grain is close to the phi slice
                phi_diff = min(abs(grain_phi - slice_value), 2*np.pi - abs(grain_phi - slice_value))
                
                if phi_diff < 0.2:  # Within tolerance
                    # Get grain property value
                    if field_property == 'awareness':
                        prop_value = grain.awareness
                    elif field_property == 'phase':
                        if hasattr(manifold, 'get_phase_continuity'):
                            prop_value = manifold.get_phase_continuity(grain_id) % (2*np.pi) / (2*np.pi)
                        else:
                            prop_value = 0.5
                    elif field_property == 'tension':
                        if hasattr(grain, 'field_tension'):
                            prop_value = grain.field_tension
                        elif hasattr(grain, 'unresolved_tension'):
                            prop_value = grain.unresolved_tension
                        else:
                            prop_value = 0.0
                    else:
                        prop_value = 0.5
                    
                    # Plot grain as point
                    ax.scatter(grain_theta, prop_value, s=100*grain.awareness, 
                              color=colormap(prop_value), edgecolors='black', alpha=0.8)
        
        # Format axes
        ax.set_xlim(0, 2*np.pi)
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        return fig, ax