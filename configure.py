"""
Enhanced Configuration Module - Centralized configuration for Collapse Geometry simulation

This module provides a unified configuration system for the Collapse Geometry
framework, eliminating redundant configuration across different components and
ensuring proper initialization of the simulation environment.
"""

import dataclasses
from typing import Dict, List, Tuple, Set, Optional, Any
import json
import os
import importlib
import sys


@dataclasses.dataclass
class CollapseGeometryConfig:
    """
    Unified configuration for the Collapse Geometry framework.
    All configuration parameters are stored here to avoid redundancy.
    """
    # Field continuity parameters
    field_continuity: float = 0.8  # How continuous fields propagate
    gradient_amplification: float = 0.3  # How much gradients are amplified
    duality_emergence: float = 0.5  # How strongly duality patterns emerge
    field_memory: float = 0.9  # How much field history influences current state
    awareness_diffusion: float = 0.2  # Rate of awareness diffusion
    resonance_amplification: float = 0.4  # How much resonance amplifies effects
    collapse_rate: float = 0.2  # Controls field evolution rate
    
    # Epistemology tensor parameters
    resolution_sensitivity: float = 0.7  # How sensitive resolution is to field consistency
    frustration_sensitivity: float = 0.4  # How sensitive frustration is to field inconsistency
    fidelity_sensitivity: float = 0.6  # How sensitive fidelity is to field memory alignment
    
    # Ancestry and memory parameters
    ancestry_coupling: float = 0.5  # How strongly ancestry influences flow
    memory_persistence: float = 0.8  # How strongly memory persists
    polarity_coupling: float = 0.25  # How memory polarity biases flow
    
    # System limits and thresholds
    min_awareness: float = 0.01  # Minimum awareness below which nodes are pruned
    backflow_threshold: float = 0.65  # Threshold for backflow to occur
    coherence_threshold: float = 0.6  # Threshold for field coherence effects
    activation_threshold: float = 0.5  # Threshold for grain activation
    
    # Field diffusion parameters
    field_diffusion_rate: float = 0.15  # Controls awareness propagation
    field_gradient_sensitivity: float = 0.25  # Sensitivity to field gradients
    
    # NEW: Toroidal dynamics parameters
    toroidal_coupling: float = 0.6        # How strongly toroidal dynamics couple to field
    phase_stability_factor: float = 0.7   # Influence of phase stability on field evolution
    major_circle_bias: float = 0.65       # Bias toward major circle dynamics (vs minor)
    toroidal_resonance_threshold: float = 0.5  # Threshold for toroidal resonance effects
    vortex_influence_radius: float = 0.3  # How far vortex effects propagate on torus
    phase_coupling_strength: float = 0.4  # How strongly phases couple between related grains
    
    # NEW: Void-Decay parameters
    void_formation_threshold: float = 0.8  # Threshold for void formation
    void_diffusion_rate: float = 0.15      # How quickly voids spread
    decay_emission_rate: float = 0.2       # How frequently decay particles emit
    decay_effect_strength: float = 0.3     # How strongly decay affects the system
    void_impact_factor: float = 0.4        # How strongly voids impact dynamics
    
    # Simulation control parameters
    max_iterations: int = 5000        # Maximum simulation steps
    snapshot_interval: int = 100      # Take state snapshot every N steps
    visualization_enabled: bool = True  # Whether to enable visualization
    log_level: str = "INFO"           # Logging level (DEBUG, INFO, WARNING, ERROR)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CollapseGeometryConfig':
        """Create configuration from dictionary"""
        # Filter out unknown keys
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)
    
    def save(self, filename: str):
        """Save configuration to file"""
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filename: str) -> 'CollapseGeometryConfig':
        """Load configuration from file"""
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Default configuration instance
default_config = CollapseGeometryConfig()


class ConfigurationManager:
    """
    Manager for handling configuration across the simulation.
    Provides a single source of truth for configuration.
    """
    
    def __init__(self, config: Optional[CollapseGeometryConfig] = None):
        self.config = config or default_config
        self._initialized_components = set()
    
    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                
        # Propagate updates to already initialized components
        self._propagate_updates()
    
    def get_config(self) -> CollapseGeometryConfig:
        """Get the current configuration"""
        return self.config
    
    def set_config(self, config: CollapseGeometryConfig):
        """Set a new configuration"""
        self.config = config
        # Propagate new config to already initialized components
        self._propagate_updates()
    
    def initialize_component(self, component: Any) -> bool:
        """
        Initialize a component with the current configuration.
        Returns True if newly initialized, False if already initialized.
        """
        if component in self._initialized_components:
            return False
        
        # Apply configuration based on component type
        if hasattr(component, 'config'):
            # Direct config attribute
            self._apply_config_to_component(component)
        elif hasattr(component, 'set_config'):
            # Component has explicit set_config method
            component.set_config(self.config)
        elif hasattr(component, 'configure'):
            # Component has configure method
            component.configure(self.config)
        elif hasattr(component, 'config_manager'):
            # Component has a config_manager
            component.config_manager.set_config(self.config)
        
        # Mark as initialized
        self._initialized_components.add(component)
        return True
    
    def _apply_config_to_component(self, component: Any):
        """Apply configuration to a component with a config attribute"""
        # Get the component's existing config
        if hasattr(component, 'config') and isinstance(component.config, dict):
            # Copy all matching attributes from global config to component config
            for field in dataclasses.fields(self.config):
                if field.name in component.config:
                    component.config[field.name] = getattr(self.config, field.name)
        elif hasattr(component, 'config'):
            # Direct attribute assignment
            component.config = self.config
    
    def _propagate_updates(self):
        """Propagate configuration updates to all initialized components"""
        for component in self._initialized_components:
            self.initialize_component(component)  # Re-initialize with new config


# Global configuration manager instance
config_manager = ConfigurationManager()


def get_config() -> CollapseGeometryConfig:
    """Get the current global configuration"""
    return config_manager.get_config()


def update_config(**kwargs) -> CollapseGeometryConfig:
    """
    Update the global configuration
    
    Returns:
        Updated configuration
    """
    config_manager.update(**kwargs)
    return config_manager.get_config()


def load_config(filename: str) -> CollapseGeometryConfig:
    """
    Load configuration from file and update global configuration
    
    Returns:
        Loaded configuration
    """
    config = CollapseGeometryConfig.load(filename)
    config_manager.set_config(config)
    return config


def save_config(filename: str):
    """Save current global configuration to file"""
    config_manager.get_config().save(filename)


def initialize_with_config(component: Any) -> bool:
    """
    Initialize a component with the current global configuration
    
    Args:
        component: The component to initialize
        
    Returns:
        True if newly initialized, False if already initialized
    """
    return config_manager.initialize_component(component)


def create_simulation():
    """
    Create a complete simulation environment including manifold, engine, and state.
    
    Returns:
        Tuple of (manifold, engine, state)
    """
    try:
        # First attempt: Direct imports from core modules
        from axiom7.core.relational_manifold import RelationalManifold
        from axiom7.core.engine import EnhancedContinuousDualityEngine
        from axiom7.core.state import SimulationState, create_simulation_state
        
        # Create components
        manifold = RelationalManifold()
        state = create_simulation_state()
        engine = EnhancedContinuousDualityEngine(manifold, state, config_manager.get_config().to_dict())
        
    except ImportError:
        try:
            # Second attempt: Try using the create_simulation from __init__.py
            import axiom7
            if hasattr(axiom7, 'create_simulation'):
                manifold, engine, state = axiom7.create_simulation()
            else:
                # Third attempt: Manual component creation
                from axiom7 import RelationalManifold, EnhancedContinuousDualityEngine, SimulationState
                
                manifold = RelationalManifold()
                state = SimulationState()
                engine = EnhancedContinuousDualityEngine(manifold, state)
        except ImportError:
            raise ImportError("Could not import required simulation components. Check that axiom7 is properly installed.")
    
    # Initialize all components with our configuration
    initialize_with_config(manifold)
    initialize_with_config(engine)
    initialize_with_config(state)
    
    return manifold, engine, state


# Predefined configurations for specific experiments
def get_stable_vortex_config() -> CollapseGeometryConfig:
    """Configuration optimized for stable vortex formation"""
    config = CollapseGeometryConfig()
    config.collapse_rate = 0.15
    config.field_continuity = 0.85
    config.gradient_amplification = 0.4
    config.field_diffusion_rate = 0.18
    config.toroidal_coupling = 0.75
    config.phase_stability_factor = 0.8
    config.major_circle_bias = 0.7
    config.vortex_influence_radius = 0.4
    return config


def get_high_complexity_config() -> CollapseGeometryConfig:
    """Configuration optimized for complex emergent behavior"""
    config = CollapseGeometryConfig()
    config.duality_emergence = 0.7
    config.awareness_diffusion = 0.3
    config.resonance_amplification = 0.6
    config.memory_persistence = 0.9
    config.ancestry_coupling = 0.7
    config.coherence_threshold = 0.5
    return config


def get_void_decay_dominant_config() -> CollapseGeometryConfig:
    """Configuration with strong void-decay dynamics"""
    config = CollapseGeometryConfig()
    config.void_formation_threshold = 0.7
    config.void_diffusion_rate = 0.2
    config.decay_emission_rate = 0.3
    config.decay_effect_strength = 0.4
    config.void_impact_factor = 0.6
    return config


def get_low_noise_config() -> CollapseGeometryConfig:
    """Configuration with low noise and high stability"""
    config = CollapseGeometryConfig()
    config.field_continuity = 0.95
    config.awareness_diffusion = 0.1
    config.field_diffusion_rate = 0.08
    config.phase_stability_factor = 0.9
    config.memory_persistence = 0.95
    return config


# Register predefined configuration generators
predefined_configs = {
    'stable_vortex': get_stable_vortex_config,
    'high_complexity': get_high_complexity_config,
    'void_decay_dominant': get_void_decay_dominant_config,
    'low_noise': get_low_noise_config
}


def get_predefined_config(config_name: str) -> Optional[CollapseGeometryConfig]:
    """
    Get a predefined configuration by name
    
    Args:
        config_name: Name of the predefined configuration
        
    Returns:
        Configuration instance or None if not found
    """
    if config_name in predefined_configs:
        return predefined_configs[config_name]()
    return None