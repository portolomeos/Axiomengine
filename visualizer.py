"""
Visualizer - Core visualization classes for Collapse Geometry

Defines the base classes for visualizing the relational manifold.
These are implemented by specific visualizers like torus simulation and unwrapping.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Any, Union


class RelationalVisualizer:
    """
    Base class for visualizing relational manifold structures.
    
    Provides common functionality for all visualizers.
    """
    
    def __init__(self):
        """Initialize the relational visualizer"""
        self.figure = None
        self.axes = None
    
    def create_figure(self, figsize=(10, 8)):
        """Create a new figure for visualization"""
        self.figure, self.axes = plt.subplots(figsize=figsize)
        return self.figure, self.axes
    
    def show(self):
        """Display the current visualization"""
        if self.figure:
            plt.show()
    
    def save(self, filename: str):
        """Save the current visualization to a file"""
        if self.figure:
            self.figure.savefig(filename, dpi=300, bbox_inches='tight')
    
    def close(self):
        """Close the current figure"""
        if self.figure:
            plt.close(self.figure)
            self.figure = None
            self.axes = None


class AnimatedVisualizer(RelationalVisualizer):
    """
    Base class for creating animated visualizations of the relational manifold.
    """
    
    def __init__(self):
        """Initialize the animated visualizer"""
        super().__init__()
        self.animation = None
    
    def create_animation(self, manifold_states, frames=20, interval=200):
        """
        Create an animation from a sequence of manifold states.
        
        Args:
            manifold_states: List of manifold states at different time points
            frames: Number of frames
            interval: Delay between frames in milliseconds
            
        Returns:
            Animation object
        """
        raise NotImplementedError("Subclasses must implement create_animation")
    
    def save_animation(self, filename: str, fps=15, dpi=300):
        """Save the animation to a file"""
        if self.animation:
            self.animation.save(filename, fps=fps, dpi=dpi)


class FieldVisualizer(RelationalVisualizer):
    """
    Base class for field-focused visualizations that emphasize continuous fields
    rather than discrete node-link structures.
    """
    
    def __init__(self):
        """Initialize the field visualizer"""
        super().__init__()
        # Define standard colormaps for various field properties
        self.colormaps = {}
        self._setup_colormaps()
    
    def _setup_colormaps(self):
        """Set up standard colormaps for field visualization"""
        # Create basic colormaps for field properties
        self.colormaps = {
            'awareness': self._create_awareness_colormap(),
            'phase': plt.cm.hsv,  # Circular colormap for phase
            'tension': plt.cm.hot,
            'polarity': plt.cm.coolwarm,
            'saturation': plt.cm.viridis
        }
    
    def _create_awareness_colormap(self):
        """Create a custom colormap for awareness visualization"""
        return mcolors.LinearSegmentedColormap.from_list(
            'awareness_cmap',
            [(0.0, '#081B41'),    # Deep dark blue (void/low awareness)
             (0.3, '#3A4FB8'),    # Medium blue (emerging awareness)
             (0.6, '#8A64D6'),    # Purple (active awareness)
             (0.8, '#E6CD87'),    # Light gold (high awareness)
             (1.0, '#FFE74C')],   # Bright gold (peak awareness)
            N=256
        )
    
    def update_state(self, manifold):
        """
        Update the visualizer state based on the current manifold state.
        
        Args:
            manifold: The relational manifold
        """
        # This should be implemented by subclasses to update internal state
        raise NotImplementedError("Subclasses must implement update_state")
    
    def visualize_field(self, manifold, property_name='awareness', **kwargs):
        """
        Visualize a specific field property of the manifold.
        
        Args:
            manifold: The relational manifold
            property_name: The property to visualize (e.g., 'awareness', 'resonance')
            **kwargs: Additional visualization parameters
            
        Returns:
            Figure and axes objects
        """
        raise NotImplementedError("Subclasses must implement visualize_field")
    
    def visualize_vector_field(self, manifold, vector_property='polarity', **kwargs):
        """
        Visualize vector field properties as arrows or streamlines.
        
        Args:
            manifold: The relational manifold
            vector_property: The vector property to visualize (e.g., 'polarity', 'momentum')
            **kwargs: Additional visualization parameters
            
        Returns:
            Figure and axes objects
        """
        raise NotImplementedError("Subclasses must implement visualize_vector_field")


class ToroidalVisualizerBase(FieldVisualizer):
    """
    Base class for toroidal visualization that provides common functionality
    for both 3D and 2D torus visualizers.
    """
    
    def __init__(self, 
                 field_resolution: int = 100, 
                 adaptation_rate: float = 0.4,
                 memory_weight: float = 0.6):
        """
        Initialize the toroidal visualizer base.
        
        Args:
            field_resolution: Resolution of field grid for visualization
            adaptation_rate: Rate at which field projections adapt to changes
            memory_weight: Influence of memory on field projection
        """
        super().__init__()
        
        # Core parameters
        self.field_resolution = field_resolution
        self.adaptation_rate = adaptation_rate
        self.memory_weight = memory_weight
        
        # Relational mapping state
        self._grain_positions = {}     # Maps grain_id -> (theta, phi) positions on torus
        self._memory_embeddings = {}   # Maps grain_id -> memory embedding (for continuity)
        self._relation_strengths = defaultdict(dict)  # Maps grain_id -> {related_id: strength}
        self._ancestry_couplings = defaultdict(set)  # Maps grain_id -> set of coupled ancestors
        
        # Enhanced colormaps for toroidal visualization
        self._setup_extended_colormaps()
    
    def _setup_extended_colormaps(self):
        """Set up specialized colormaps for toroidal field visualization"""
        # Add additional colormaps specific to toroidal visualization
        self.colormaps.update({
            'void': self._create_void_colormap(),
            'vortex': self._create_vortex_colormap(),
            'transition': self._create_transition_colormap()
        })
    
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
        
        Args:
            manifold: The relational manifold
        """
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
                self._memory_embeddings[grain_id] = np.random.random(4)  # 4D embedding
            
            memory_vec = self._memory_embeddings[grain_id]
            
            # Project memory to toroidal coordinates
            memory_theta = np.arctan2(memory_vec[1], memory_vec[0]) % (2*np.pi)
            memory_phi = np.arctan2(memory_vec[3], memory_vec[2]) % (2*np.pi)
            
            # Current position
            current_theta, current_phi = self._grain_positions[grain_id]
            
            # Calculate circular differences
            d_theta = np.sin(memory_theta - current_theta)
            d_phi = np.sin(memory_phi - current_phi)
            
            # Update based on memory influence
            position_updates[grain_id][0] += d_theta * 0.2
            position_updates[grain_id][1] += d_phi * 0.2
        
        # Apply updates with toroidal wrapping
        for grain_id in current_grains:
            current_theta, current_phi = self._grain_positions[grain_id]
            
            # Update with adaptation rate
            new_theta = (current_theta + position_updates[grain_id][0] * self.adaptation_rate) % (2*np.pi)
            new_phi = (current_phi + position_updates[grain_id][1] * self.adaptation_rate) % (2*np.pi)
            
            self._grain_positions[grain_id] = (new_theta, new_phi)
    
    def get_grain_property(self, grain, property_name, default=0.0):
        """
        Helper method to safely get grain properties.
        
        Args:
            grain: The grain object
            property_name: Name of the property to get
            default: Default value if property is not found
            
        Returns:
            Property value or default
        """
        if hasattr(grain, property_name):
            return getattr(grain, property_name)
        
        # Try nested properties (e.g., 'field_polarity.theta')
        if '.' in property_name:
            parts = property_name.split('.')
            current = grain
            
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    return default
            
            return current
        
        return default