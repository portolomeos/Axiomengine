"""
Collapse Rules Package - Rules for configuration, polarity and emergent behavior

This package contains the rules that govern how collapse occurs in the simulation,
how fields interact, and how emergent phenomena form.
Enhanced with Void-Decay principle for handling incompatible structures and
toroidal referencing for richer topological dynamics.
"""

# Use lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'Grain':
        from axiom7.collapse_rules.grain_dynamics import Grain
        return Grain
    elif name == 'ToroidalGrainSystem':
        from axiom7.collapse_rules.grain_dynamics import ToroidalGrainSystem
        return ToroidalGrainSystem
    elif name == 'ConfigurationPoint':
        from axiom7.collapse_rules.config_space import ConfigurationPoint
        return ConfigurationPoint
    elif name == 'ConfigurationSpace':
        from axiom7.collapse_rules.config_space import ConfigurationSpace
        return ConfigurationSpace
    elif name == 'create_configuration_space':
        from axiom7.collapse_rules.config_space import create_configuration_space
        return create_configuration_space
    elif name == 'EpistomologyRelation':
        from axiom7.collapse_rules.polarity_space import EpistomologyRelation
        return EpistomologyRelation
    elif name == 'EpistomologyField':
        from axiom7.collapse_rules.polarity_space import EpistomologyField
        return EpistomologyField
    elif name == 'RelativeRotationTensor':
        from axiom7.collapse_rules.polarity_space import RelativeRotationTensor
        return RelativeRotationTensor
    elif name == 'EmergentFieldRule':
        from axiom7.collapse_rules.emergent_field_rules import EmergentFieldRule
        return EmergentFieldRule
    elif name == 'EnhancedPhaseClassificationRule':
        from axiom7.collapse_rules.emergent_field_rules import EnhancedPhaseClassificationRule
        return EnhancedPhaseClassificationRule
    elif name == 'ToroidalStructureRule':
        from axiom7.collapse_rules.emergent_field_rules import ToroidalStructureRule
        return ToroidalStructureRule
    elif name == 'VoidDecayRule':
        from axiom7.collapse_rules.emergent_field_rules import VoidDecayRule
        return VoidDecayRule
    elif name == 'AncestryEntanglementRule':
        from axiom7.collapse_rules.emergent_field_rules import AncestryEntanglementRule
        return AncestryEntanglementRule
    elif name == 'RecurrencePatternRule':
        from axiom7.collapse_rules.emergent_field_rules import RecurrencePatternRule
        return RecurrencePatternRule
    elif name == 'create_field_rule':
        from axiom7.collapse_rules.emergent_field_rules import create_field_rule
        return create_field_rule
    elif name == 'toroidal_distance':
        from axiom7.collapse_rules.grain_dynamics import toroidal_distance
        return toroidal_distance
    elif name == 'angular_difference':
        from axiom7.collapse_rules.grain_dynamics import angular_difference
        return angular_difference
    elif name == 'circular_mean':
        from axiom7.collapse_rules.grain_dynamics import circular_mean
        return circular_mean
    else:
        raise AttributeError(f"module 'axiom7.collapse_rules' has no attribute '{name}'")

# Helper functions
def create_random_grain(grain_id: str = None, toroidal: bool = True):
    """
    Create a random grain with optional toroidal properties.
    
    Args:
        grain_id: Optional grain ID (random UUID if None)
        toroidal: Whether to include toroidal properties
    
    Returns:
        New Grain instance
    """
    from axiom7.collapse_rules.grain_dynamics import create_random_grain
    return create_random_grain(grain_id)

def create_toroidal_grain_system(neighborhood_radius: float = 0.5):
    """
    Create a new toroidal grain system.
    
    Args:
        neighborhood_radius: Radius for neighborhood detection
        
    Returns:
        New ToroidalGrainSystem instance
    """
    from axiom7.collapse_rules.grain_dynamics import ToroidalGrainSystem
    return ToroidalGrainSystem(neighborhood_radius=neighborhood_radius)

# Export the key classes and functions
__all__ = [
    'Grain',
    'ToroidalGrainSystem',
    'ConfigurationPoint',
    'ConfigurationSpace',
    'create_configuration_space',
    'EpistomologyRelation',
    'EpistomologyField',
    'RelativeRotationTensor',
    'EmergentFieldRule',
    'EnhancedPhaseClassificationRule',
    'ToroidalStructureRule',
    'VoidDecayRule',
    'AncestryEntanglementRule',
    'RecurrencePatternRule',
    'create_field_rule',
    'create_random_grain',
    'create_toroidal_grain_system',
    'toroidal_distance',
    'angular_difference',
    'circular_mean'
]