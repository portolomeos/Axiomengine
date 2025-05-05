"""
Enhanced Continuous Duality Engine - Pure Relational Implementation with Toroidal Dynamics

A redesigned engine implementing the integrated collapse geometry approach where:
1. Everything is relational: structure emerges from local interactions, not external rules
2. Collapse is the only source of committed structure
3. Awareness fields evolve through relational tension, not imposed trajectories
4. Polarity bias represents directional memory, not force
5. Ancestry is a geometric memory field: shared ancestry defines entanglement
6. Local coherence, memory, and ancestry shape future flow and structure recursively
7. The engine observes; the field *is* the system
8. Curvature is defined by how flow and memory reinforce or bend each other over time

Enhanced with Void-Decay principle for handling incompatible structures and
toroidal referencing for richer topological dynamics.
"""

import random
import math
from typing import Dict, List, Tuple, Set, Optional, Callable, Any, TYPE_CHECKING, Union

# Direct imports that don't cause circular dependencies
from axiom7.collapse_rules.emergent_field_rules import create_field_rule
from axiom7.collapse_rules.grain_dynamics import angular_difference, toroidal_distance, circular_mean

# Use TYPE_CHECKING to prevent circular imports at runtime
if TYPE_CHECKING:
    from axiom7.core.relational_manifold import RelationalManifold
    from axiom7.core.state import SimulationState
    from axiom7.collapse_rules.grain_dynamics import Grain
    from axiom7.collapse_rules.emergent_field_rules import EmergentFieldRule, RecurrencePatternRule
    from axiom7.collapse_rules.emergent_field_rules import ToroidalStructureRule

class EnhancedContinuousDualityEngine:
    """
    Pure relational simulation engine with an integrated approach where all dynamics 
    emerge directly from relational properties without fixed coordinates.
    
    Field and manifold are unified - the field is a continuous view of the manifold,
    not a separate component.
    
    Enhanced with:
    - Void-Decay support for handling incompatible structures
    - Toroidal referencing for proper topological dynamics
    """
    
    def __init__(self, manifold=None, state=None, config=None):
        """
        Initialize the continuous duality engine.
        
        Args:
            manifold: The relational manifold to operate on
            state: The simulation state to track
            config: Optional configuration dictionary to override defaults
        """
        # Import here to prevent circular imports
        if manifold is None:
            from axiom7.core.relational_manifold import RelationalManifold
            self.manifold = RelationalManifold()
        else:
            self.manifold = manifold
            
        # Import state if needed
        if state is None:
            from axiom7.core.state import SimulationState
            self.state = SimulationState()
        else:
            self.state = state
        
        # Configuration parameters - all controlling relational dynamics
        self.config = {
            # Field continuity parameters
            'field_continuity': 0.8,        # How continuous fields propagate
            'gradient_amplification': 0.3,  # How much gradients are amplified
            'duality_emergence': 0.5,       # How strongly duality patterns emerge
            'field_memory': 0.9,            # How much field history influences current state
            'awareness_diffusion': 0.2,     # Rate of awareness diffusion
            'resonance_amplification': 0.4, # How much resonance amplifies effects
            'collapse_rate': 0.2,           # Controls field evolution rate
            
            # Epistemology tensor parameters
            'resolution_sensitivity': 0.7,  # How sensitive resolution is to field consistency
            'frustration_sensitivity': 0.4, # How sensitive frustration is to field inconsistency
            'fidelity_sensitivity': 0.6,    # How sensitive fidelity is to field memory alignment
            
            # Ancestry and memory parameters
            'ancestry_coupling': 0.5,       # How strongly ancestry influences flow
            'memory_persistence': 0.8,      # How strongly memory persists
            'polarity_coupling': 0.25,      # How memory polarity biases flow
            
            # Void-Decay parameters
            'void_formation_threshold': 0.8, # Threshold for void formation
            'void_diffusion_rate': 0.15,     # How quickly voids spread
            'decay_emission_rate': 0.2,      # How frequently decay particles emit
            'decay_effect_strength': 0.3,    # How strongly decay affects the system
            'void_impact_factor': 0.4,       # How strongly voids impact dynamics
            
            # NEW: Toroidal dynamics parameters
            'toroidal_coupling': 0.6,        # How strongly toroidal dynamics couple to field
            'phase_stability_factor': 0.7,   # Influence of phase stability on field evolution
            'major_circle_bias': 0.65,       # Bias toward major circle dynamics (vs minor)
            'toroidal_resonance_threshold': 0.5, # Threshold for toroidal resonance effects
            'vortex_influence_radius': 0.3,  # How far vortex effects propagate on torus
            'phase_coupling_strength': 0.4,  # How strongly phases couple between related grains
            
            # System limits
            'min_awareness': 0.01,          # Minimum awareness below which nodes are pruned
            'backflow_threshold': 0.65,     # Threshold for backflow to occur
            'coherence_threshold': 0.6      # Threshold for field coherence effects
        }
        
        # Override with provided config
        if config:
            self.config.update(config)
        
        # Sync configuration with manifold for shared parameters
        self._sync_manifold_config()
        
        # Initialize event handlers
        self.event_handlers = {
            'on_collapse': [],
            'on_field_update': [],
            'on_step': [],
            'on_backflow': [],
            'on_negation': [],
            'on_opposition': [],
            'on_continuous_flow': [],
            'on_field_resonance': [],
            'on_pattern_emergence': [],
            'on_void_formation': [],           # Void-Decay events
            'on_decay_emission': [],
            'on_incompatible_structure': [],
            'on_toroidal_vortex': [],          # NEW: Toroidal-specific events
            'on_phase_transition': [],
            'on_toroidal_resonance': [],
            'on_toroidal_domain_formation': []
        }
        
        # NEW: Initialize toroidal dynamics tracking
        self.toroidal_metrics = {
            'phase_coherence': 0.0,
            'domain_count': 0,
            'vortex_count': 0,
            'major_circle_flow': 0.0,
            'minor_circle_flow': 0.0
        }
        self._last_toroidal_check_time = 0.0
    
    def _sync_manifold_config(self):
        """
        Synchronize configuration with manifold to eliminate redundancy.
        Maps engine config parameters to corresponding manifold parameters.
        Enhanced with toroidal configuration parameters.
        """
        # Map of engine config keys to manifold attributes
        config_map = {
            'field_diffusion_rate': 'field_diffusion_rate',
            'field_gradient_sensitivity': 'field_gradient_sensitivity',
            'activation_threshold': 'activation_threshold'
        }
        
        # Update manifold attributes
        for engine_key, manifold_attr in config_map.items():
            if engine_key in self.config and hasattr(self.manifold, manifold_attr):
                setattr(self.manifold, manifold_attr, self.config[engine_key])
        
        # Sync void-decay parameters if supported
        if hasattr(self.manifold, 'void_decay_config'):
            void_decay_map = {
                'void_formation_threshold': 'void_formation_threshold',
                'decay_emission_rate': 'decay_emission_rate',
                'void_diffusion_rate': 'void_propagation_rate',
                'decay_effect_strength': 'decay_impact_factor'
            }
            
            for engine_key, manifold_key in void_decay_map.items():
                if engine_key in self.config:
                    self.manifold.void_decay_config[manifold_key] = self.config[engine_key]
        
        # NEW: Sync toroidal parameters if supported
        if hasattr(self.manifold, 'toroidal_system'):
            # Set neighborhood radius based on vortex influence radius
            if hasattr(self.manifold.toroidal_system, 'neighborhood_radius'):
                self.manifold.toroidal_system.neighborhood_radius = self.config['vortex_influence_radius']
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register a callback function for a specific event type"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
    
    def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """
        Trigger event handlers and log to state in one operation.
        This eliminates the redundancy between event triggering and logging.
        """
        # Call handlers
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                handler(data)
        
        # Log to state if available
        if self.state:
            # Add time if not present
            if 'time' not in data:
                data['time'] = self.manifold.time
            
            # Convert event type if needed for state compatibility
            state_event_type = event_type.replace('on_', '')
            self.state.log_event(state_event_type, data)
    
    def _ensure_epistemology_support(self):
        """
        Ensure nodes/grains support the epistemology approach by checking for required methods.
        If not supported, this will be handled gracefully with backward compatibility.
        
        Returns:
            bool: True if epistemology is supported, False otherwise
        """
        # Check if manifold uses grains instead of nodes
        if hasattr(self.manifold, 'grains') and not hasattr(self.manifold, 'nodes'):
            # Use grains collection for manifolds that use grains
            sample_grain = next(iter(self.manifold.grains.values()), None)
            if not sample_grain:
                return False  # No grains to check
            
            # Check for basic epistemology tensor support
            has_epistemology = hasattr(sample_grain, 'relation_epistemology')
            has_update_method = hasattr(sample_grain, 'update_relation_epistemology')
            
            return has_epistemology and has_update_method
        else:
            # Original implementation for manifolds that use nodes
            sample_node = next(iter(getattr(self.manifold, 'nodes', {}).values()), None)
            if not sample_node:
                return False  # No nodes to check
            
            # Check for basic epistemology tensor support
            has_epistemology = hasattr(sample_node, 'relation_epistemology')
            has_update_method = hasattr(sample_node, 'update_relation_epistemology')
            
            return has_epistemology and has_update_method
    
    def _ensure_toroidal_support(self):
        """
        NEW: Ensure manifold supports toroidal dynamics by checking for required methods.
        """
        # Check for basic toroidal support in manifold
        has_toroidal_system = hasattr(self.manifold, 'toroidal_system')
        has_toroidal_phase = hasattr(self.manifold, 'toroidal_phase')
        has_get_toroidal_phase = hasattr(self.manifold, 'get_toroidal_phase')
        
        return has_toroidal_system and has_toroidal_phase and has_get_toroidal_phase
    
    def process_continuous_fields(self):
        """
        Process all fields as continuous processes using epistemology components.
        All fields evolve across the relational network based on gradients, resonance, and memory.
        Enhanced with void-decay handling and toroidal dynamics.
        """
        # Process fields directly in the manifold
        self.manifold.update_continuous_fields()
        
        # Process patterns in a safer way
        self._process_emergent_patterns()
        
        # Process emergent duality from field patterns
        self._process_emergent_duality()
        
        # Process void and decay effects
        self._process_void_decay_effects()
        
        # NEW: Process toroidal dynamics
        self._process_toroidal_dynamics()
        
        # Trigger field update event
        self._trigger_event('on_field_update', {
            'time': self.manifold.time,
            'update_type': 'continuous_field'
        })
    
    def _process_emergent_patterns(self):
        """
        Process emergent patterns using field rules.
        Enhanced with toroidal pattern detection.
        """
        try:
            # Create recurrence pattern rule
            recurrence_rule = create_field_rule('recurrence')
            
            # Evaluate patterns
            pattern_results = recurrence_rule.evaluate(self.manifold)
            
            # Process patterns if available
            if 'patterns' in pattern_results and isinstance(pattern_results['patterns'], list):
                for pattern in pattern_results['patterns']:
                    if isinstance(pattern, dict):
                        pattern_type = pattern.get('type', '')
                        
                        if 'resonant' in pattern_type:
                            # Trigger field resonance event
                            self._trigger_event('on_field_resonance', {
                                'pattern_nodes': pattern.get('grains', []),
                                'resonance': pattern.get('strength', 0.0),
                                'time': self.manifold.time
                            })
                        elif 'collapse' in pattern_type or 'cycle' in pattern_type:
                            # Trigger continuous flow event
                            self._trigger_event('on_continuous_flow', {
                                'pattern_nodes': [pattern.get('grain_id', '')],
                                'pattern': pattern.get('pattern', []),
                                'time': self.manifold.time
                            })
                        elif 'vortex' in pattern_type:
                            # Trigger vortex pattern event
                            self._trigger_event('on_pattern_emergence', {
                                'pattern_type': 'vortex',
                                'center_grain': pattern.get('center_grain', ''),
                                'strength': pattern.get('strength', 0.0),
                                'time': self.manifold.time
                            })
                        elif 'toroidal_flow' in pattern_type:
                            # NEW: Trigger toroidal flow pattern event
                            self._trigger_event('on_pattern_emergence', {
                                'pattern_type': 'toroidal_flow',
                                'flow_direction': pattern.get('flow_direction', ''),
                                'flow_type': pattern.get('flow_type', ''),
                                'strength': pattern.get('strength', 0.0),
                                'size': pattern.get('size', 0),
                                'time': self.manifold.time
                            })
                        else:
                            # General pattern emergence event
                            self._trigger_event('on_pattern_emergence', {
                                'pattern_type': pattern_type,
                                'nodes': pattern.get('grains', []), 
                                'time': self.manifold.time,
                                'properties': pattern
                            })
            
            # NEW: Try to use toroidal structure rule directly
            try:
                # Create toroidal structure rule
                toroidal_rule = create_field_rule('toroidal')
                
                # Evaluate toroidal patterns
                toroidal_results = toroidal_rule.evaluate(self.manifold)
                
                # Update toroidal metrics
                if 'dominant_theta_mode' in toroidal_results:
                    self.toroidal_metrics['dominant_theta_mode'] = toroidal_results['dominant_theta_mode']
                if 'dominant_phi_mode' in toroidal_results:
                    self.toroidal_metrics['dominant_phi_mode'] = toroidal_results['dominant_phi_mode']
                if 'major_circle_flow' in toroidal_results:
                    self.toroidal_metrics['major_circle_flow'] = toroidal_results['major_circle_flow']
                if 'minor_circle_flow' in toroidal_results:
                    self.toroidal_metrics['minor_circle_flow'] = toroidal_results['minor_circle_flow']
                
                # Process cross-phase structures
                if 'cross_phase_structures' in toroidal_results:
                    for structure in toroidal_results['cross_phase_structures']:
                        # Trigger toroidal resonance event
                        self._trigger_event('on_toroidal_resonance', {
                            'theta_mode': structure.get('theta_mode', 0),
                            'phi_mode': structure.get('phi_mode', 0),
                            'ratio': structure.get('ratio', 0),
                            'type': structure.get('type', ''),
                            'strength': structure.get('strength', 0.0),
                            'time': self.manifold.time
                        })
            except (ImportError, AttributeError, TypeError) as e:
                # Fall back to basic toroidal detection if advanced methods fail
                pass
                
            # NEW: Try to use phase classification rule for phase domains
            try:
                # Create phase classification rule
                phase_rule = create_field_rule('phase_classification')
                
                # Evaluate phase patterns
                phase_results = phase_rule.evaluate(self.manifold)
                
                # Process phase domains
                if 'phase_domains' in phase_results:
                    # Update domain count in metrics
                    self.toroidal_metrics['domain_count'] = len(phase_results['phase_domains'])
                    
                    for domain in phase_results['phase_domains']:
                        # Trigger domain formation event
                        self._trigger_event('on_toroidal_domain_formation', {
                            'domain_id': domain.get('domain_id', ''),
                            'phase_type': domain.get('phase_type', ''),
                            'size': len(domain.get('points', [])),
                            'stability': domain.get('stability', 0.0),
                            'theta_center': domain.get('theta_center', 0.0),
                            'phi_center': domain.get('phi_center', 0.0),
                            'time': self.manifold.time
                        })
            except (ImportError, AttributeError, TypeError) as e:
                # Gracefully handle missing phase classification
                pass
                
        except (ImportError, AttributeError) as e:
            # Fall back to direct pattern detection if field rules fail
            self._process_patterns_directly()
    
    def _process_patterns_directly(self):
        """
        Fallback method to detect patterns directly when field rules are unavailable.
        Enhanced with basic toroidal pattern detection.
        """
        # Check for vortices directly
        if hasattr(self.manifold, 'detect_vortices'):
            vortices = self.manifold.detect_vortices()
            
            # Process vortices
            for vortex in vortices:
                center_id = vortex.get('center_node', '')
                
                # Trigger vortex event
                self._trigger_event('on_pattern_emergence', {
                    'pattern_type': 'vortex',
                    'center_grain': center_id,
                    'strength': vortex.get('strength', 0.0),
                    'rotation_direction': vortex.get('rotation_direction', ''),
                    'time': self.manifold.time
                })
                
                # NEW: Also trigger toroidal vortex event if toroidal properties available
                if 'theta' in vortex and 'phi' in vortex:
                    self._trigger_event('on_toroidal_vortex', {
                        'center_grain': center_id,
                        'theta': vortex.get('theta', 0.0),
                        'phi': vortex.get('phi', 0.0),
                        'pattern_type': vortex.get('pattern_type', 'mixed'),
                        'theta_curvature': vortex.get('theta_curvature', 0.0),
                        'phi_curvature': vortex.get('phi_curvature', 0.0),
                        'time': self.manifold.time
                    })
    
    def _process_void_decay_effects(self):
        """
        Process void and decay effects across the system.
        Handles void formation, decay emissions, and their impacts.
        """
        # Check if manifold supports void-decay principles
        if not hasattr(self.manifold, 'process_void_and_decay'):
            return
        
        # Get void regions before processing
        prev_void_regions = self.manifold.find_void_regions() if hasattr(self.manifold, 'find_void_regions') else []
        prev_void_count = len(prev_void_regions)
        
        # Process void and decay
        self.manifold.process_void_and_decay()
        
        # Get updated void regions after processing
        current_void_regions = self.manifold.find_void_regions() if hasattr(self.manifold, 'find_void_regions') else []
        
        # Check for new void formations
        if len(current_void_regions) > prev_void_count:
            # New voids have formed
            new_voids = current_void_regions[prev_void_count:]
            
            for void_region in new_voids:
                # Trigger void formation event
                self._trigger_event('on_void_formation', {
                    'void_id': void_region.get('id', ''),
                    'center_id': void_region.get('center', ''),
                    'strength': void_region.get('strength', 0.0),
                    'affected_count': len(void_region.get('affected_grains', [])),
                    'time': self.manifold.time
                })
        
        # Check for decay emissions
        if hasattr(self.manifold, 'decay_emission_events'):
            # Get only new decay emissions since last processing
            last_processed_time = getattr(self, '_last_decay_process_time', 0.0)
            new_emissions = [
                event for event in self.manifold.decay_emission_events 
                if event['time'] > last_processed_time
            ]
            
            # Update last processed time
            self._last_decay_process_time = self.manifold.time
            
            # Trigger events for new emissions
            for emission in new_emissions:
                event_data = {
                    'origin_id': emission.get('origin_id', ''),
                    'strength': emission.get('strength', 0.0),
                    'memory_trace': emission.get('memory_trace', []),
                    'time': emission.get('time', self.manifold.time)
                }
                
                # NEW: Add toroidal position if available
                if 'theta' in emission and 'phi' in emission:
                    event_data['theta'] = emission['theta']
                    event_data['phi'] = emission['phi']
                
                self._trigger_event('on_decay_emission', event_data)
        
        # Check for incompatible structure events
        if hasattr(self.manifold, 'incompatible_structure_events'):
            # Get only new incompatible structure events
            last_processed_time = getattr(self, '_last_incompatible_process_time', 0.0)
            new_events = [
                event for event in self.manifold.incompatible_structure_events 
                if event['time'] > last_processed_time
            ]
            
            # Update last processed time
            self._last_incompatible_process_time = self.manifold.time
            
            # Trigger events for new incompatible structure events
            for event in new_events:
                self._trigger_event('on_incompatible_structure', {
                    'source': event.get('source', ''),
                    'target': event.get('target', ''),
                    'collapse_strength': event.get('collapse_strength', 0.0),
                    'resolution': event.get('resolution', 'unknown'),
                    'time': event.get('time', self.manifold.time)
                })
    
    def _process_toroidal_dynamics(self):
        """
        NEW: Process toroidal dynamics across the system.
        Analyzes phase coherence, vortices, and phase domains.
        """
        # Check if manifold supports toroidal dynamics
        if not self._ensure_toroidal_support():
            return
        
        # Check if enough time has passed since last toroidal check
        time_since_last_check = self.manifold.time - getattr(self, '_last_toroidal_check_time', 0.0)
        if time_since_last_check < 1.0:  # Only check periodically
            return
        
        # Update last check time
        self._last_toroidal_check_time = self.manifold.time
        
        # Get toroidal metrics from manifold if available
        if hasattr(self.manifold, 'toroidal_metrics'):
            # Update engine metrics from manifold
            for key, value in self.manifold.toroidal_metrics.items():
                if key in self.toroidal_metrics:
                    self.toroidal_metrics[key] = value
        
        # Calculate phase coherence
        if hasattr(self.manifold, 'calculate_phase_coherence'):
            coherence = self.manifold.calculate_phase_coherence()
            self.toroidal_metrics['phase_coherence'] = coherence
            
            # Check for phase transitions (significant changes in coherence)
            prev_coherence = getattr(self, '_previous_coherence', coherence)
            coherence_change = abs(coherence - prev_coherence)
            
            if coherence_change > 0.2:  # Significant change in coherence
                # Determine transition type
                transition_type = 'ordering' if coherence > prev_coherence else 'disordering'
                
                # Trigger phase transition event
                self._trigger_event('on_phase_transition', {
                    'previous_coherence': prev_coherence,
                    'new_coherence': coherence,
                    'change': coherence_change,
                    'transition_type': transition_type,
                    'time': self.manifold.time
                })
            
            # Store current coherence for next comparison
            self._previous_coherence = coherence
        
        # Check for toroidal vortices
        if hasattr(self.manifold, 'detect_vortices'):
            vortices = self.manifold.detect_vortices()
            
            # Check for new vortices
            prev_vortex_count = self.toroidal_metrics.get('vortex_count', 0)
            current_vortex_count = len(vortices)
            
            # Update vortex count in metrics
            self.toroidal_metrics['vortex_count'] = current_vortex_count
            
            # Trigger events for major vortices
            for vortex in vortices:
                # Only trigger events for strong vortices
                if vortex.get('strength', 0.0) > self.config['toroidal_resonance_threshold']:
                    self._trigger_event('on_toroidal_vortex', {
                        'center_grain': vortex.get('center_node', ''),
                        'strength': vortex.get('strength', 0.0),
                        'theta': vortex.get('theta', 0.0),
                        'phi': vortex.get('phi', 0.0),
                        'pattern_type': vortex.get('pattern_type', 'mixed'),
                        'rotation_direction': vortex.get('rotation_direction', ''),
                        'time': self.manifold.time
                    })
        
        # Find toroidal clusters
        if hasattr(self.manifold, 'find_toroidal_clusters'):
            clusters = self.manifold.find_toroidal_clusters()
            
            # Process significant clusters
            for cluster in clusters:
                # Only process large, stable clusters
                size = cluster.get('size', 0)
                if size >= 5:  # Significant cluster size
                    self._trigger_event('on_pattern_emergence', {
                        'pattern_type': 'toroidal_cluster',
                        'center_theta': cluster.get('center_theta', 0.0),
                        'center_phi': cluster.get('center_phi', 0.0),
                        'size': size,
                        'avg_awareness': cluster.get('avg_awareness', 0.0),
                        'time': self.manifold.time
                    })
    
    def _process_emergent_duality(self):
        """
        Process emergent duality patterns from field behavior.
        Duality naturally emerges from relational properties and epistemology.
        Enhanced with void awareness and toroidal opposition.
        """
        # Skip if duality emergence is disabled
        if self.config['duality_emergence'] < 0.1:
            return
            
        has_epistemology = self._ensure_epistemology_support()
        has_toroidal = self._ensure_toroidal_support()
        
        # Identify candidate opposite pairs
        opposite_candidates = []
        
        # Determine which collection to use (grains or nodes)
        node_collection = getattr(self.manifold, 'grains', {})
        if not node_collection and hasattr(self.manifold, 'nodes'):
            node_collection = self.manifold.nodes
        
        # For nodes with epistemology: find pairs with opposing epistemology patterns
        if has_epistemology:
            for node1_id, node1 in node_collection.items():
                # Skip if already has opposite
                if node1.opposite_state:
                    continue
                    
                for node2_id, node2 in node_collection.items():
                    # Skip self or nodes that already have opposites
                    if node1_id == node2_id or node2.opposite_state:
                        continue
                        
                    # Check for opposing epistemology patterns
                    # (strong memory in opposite directions)
                    opposition_score = 0.0
                    
                    # Check direct opposition in epistemology
                    if (node2_id in node1.relation_epistemology and 
                        node1_id in node2.relation_epistemology):
                        
                        # Get epistemology components
                        strength1, resolution1, frustration1, fidelity1 = node1.relation_epistemology[node2_id]
                        strength2, resolution2, frustration2, fidelity2 = node2.relation_epistemology[node1_id]
                        
                        # Opposing directions (opposite strength signs)
                        if strength1 * strength2 < 0:
                            # Opposing frustration-resolution patterns
                            if (frustration1 > resolution1 and frustration2 > resolution2):
                                opposition_score = (
                                    abs(frustration1 - resolution1) + 
                                    abs(frustration2 - resolution2)
                                ) / 2.0
                    
                    # NEW: Check for toroidal phase opposition
                    if has_toroidal:
                        # Get toroidal phases
                        theta1, phi1 = self.manifold.get_toroidal_phase(node1_id)
                        theta2, phi2 = self.manifold.get_toroidal_phase(node2_id)
                        
                        # Calculate phase opposition
                        # Maximum opposition is at π radians (180°)
                        theta_opposition = abs(angular_difference(theta1, theta2) - math.pi) / math.pi
                        phi_opposition = abs(angular_difference(phi1, phi2) - math.pi) / math.pi
                        
                        # Convert to opposition score (0 = different phase, 1 = opposite phase)
                        phase_opposition = 1.0 - (theta_opposition * 0.6 + phi_opposition * 0.4)
                        
                        # Add to total opposition score
                        opposition_score += phase_opposition * 0.4
                    
                    # Check for void presence which can enhance duality emergence
                    node1_void = self.manifold.get_void_presence(node1_id) if hasattr(self.manifold, 'get_void_presence') else 0.0
                    node2_void = self.manifold.get_void_presence(node2_id) if hasattr(self.manifold, 'get_void_presence') else 0.0
                    
                    if node1_void > 0.2 or node2_void > 0.2:
                        # Void enhances opposition
                        void_factor = max(node1_void, node2_void) * 0.5
                        opposition_score += void_factor
                    
                    # If opposition score is significant, add as candidate
                    if opposition_score > 0.3:
                        opposite_candidates.append((node1_id, node2_id, opposition_score))
        else:
            # Without epistemology, fall back to awareness, flow tendency, and toroidal phase differences
            for node1_id, node1 in node_collection.items():
                # Skip if already has opposite
                if node1.opposite_state:
                    continue
                    
                for node2_id, node2 in node_collection.items():
                    # Skip self or nodes that already have opposites
                    if node1_id == node2_id or node2.opposite_state:
                        continue
                        
                    # Check opposing flow tendencies
                    opposition_score = 0.0
                    
                    # Basic opposition score from awareness difference
                    awareness_diff = abs(node1.awareness - node2.awareness)
                    
                    if awareness_diff > 0.5:  # Significant difference
                        opposition_score += awareness_diff * 0.5
                    
                    # Check for opposing flow tendencies if method available
                    if hasattr(self.manifold, 'get_flow_tendency'):
                        flow_tendencies1 = self.manifold.get_flow_tendency(node1_id)
                        flow_tendencies2 = self.manifold.get_flow_tendency(node2_id)
                        
                        if flow_tendencies1 and flow_tendencies2:
                            # Find common relations
                            common_relations = set(flow_tendencies1.keys()) & set(flow_tendencies2.keys())
                            
                            if common_relations:
                                # Count opposing flow tendencies
                                opposing_count = 0
                                for rel_id in common_relations:
                                    flow1 = flow_tendencies1[rel_id]
                                    flow2 = flow_tendencies2[rel_id]
                                    
                                    # Opposing flow directions
                                    if flow1 * flow2 < 0:
                                        opposing_count += 1
                                
                                # Stronger opposition with more opposing flows
                                if opposing_count > 0:
                                    opposition_score += 0.5 * (opposing_count / len(common_relations))
                    
                    # NEW: Check for toroidal phase opposition
                    if has_toroidal:
                        # Get toroidal phases
                        theta1, phi1 = self.manifold.get_toroidal_phase(node1_id)
                        theta2, phi2 = self.manifold.get_toroidal_phase(node2_id)
                        
                        # Calculate phase opposition
                        # Maximum opposition is at π radians (180°)
                        theta_opposition = abs(angular_difference(theta1, theta2) - math.pi) / math.pi
                        phi_opposition = abs(angular_difference(phi1, phi2) - math.pi) / math.pi
                        
                        # Convert to opposition score (0 = different phase, 1 = opposite phase)
                        phase_opposition = 1.0 - (theta_opposition * 0.6 + phi_opposition * 0.4)
                        
                        # Add to total opposition score
                        opposition_score += phase_opposition * 0.4
                    
                    # Check for void presence which can enhance duality emergence
                    node1_void = self.manifold.get_void_presence(node1_id) if hasattr(self.manifold, 'get_void_presence') else 0.0
                    node2_void = self.manifold.get_void_presence(node2_id) if hasattr(self.manifold, 'get_void_presence') else 0.0
                    
                    if node1_void > 0.2 or node2_void > 0.2:
                        # Void enhances opposition
                        void_factor = max(node1_void, node2_void) * 0.5
                        opposition_score += void_factor
                    
                    if opposition_score > 0.3:
                        opposite_candidates.append((node1_id, node2_id, opposition_score))
        
        # Sort candidates by opposition score
        opposite_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Process top candidates
        for node1_id, node2_id, score in opposite_candidates[:5]:  # Limit to top 5
            # Create opposition with probability based on score and duality_emergence parameter
            if random.random() < score * self.config['duality_emergence']:
                # Check if set_opposite_grains or set_opposite_nodes method exists
                if hasattr(self.manifold, 'set_opposite_grains'):
                    self.manifold.set_opposite_grains(node1_id, node2_id)
                elif hasattr(self.manifold, 'set_opposite_nodes'):
                    self.manifold.set_opposite_nodes(node1_id, node2_id)
                else:
                    # Set opposites directly in the grains/nodes
                    if node1_id in node_collection and node2_id in node_collection:
                        node1 = node_collection[node1_id]
                        node2 = node_collection[node2_id]
                        node1.opposite_state = node2_id
                        node2.opposite_state = node1_id
                
                # Update epistemology if supported
                if has_epistemology:
                    node1 = self.manifold.get_grain(node1_id) if hasattr(self.manifold, 'get_grain') else self.manifold.get_node(node1_id)
                    node2 = self.manifold.get_grain(node2_id) if hasattr(self.manifold, 'get_grain') else self.manifold.get_node(node2_id)
                    
                    if node1 and node2 and hasattr(node1, 'update_relation_epistemology'):
                        # Set high frustration, moderate resolution for opposites
                        node1.update_relation_epistemology(
                            node2_id,
                            strength=-score,  # Negative strength for opposition
                            resolution=0.4,
                            frustration=0.7,
                            fidelity=0.5,
                            blending_factor=0.3
                        )
                        
                        node2.update_relation_epistemology(
                            node1_id,
                            strength=-score,  # Negative strength for opposition
                            resolution=0.4,
                            frustration=0.7,
                            fidelity=0.5,
                            blending_factor=0.3
                        )
                
                # Set tension for both nodes
                node1 = self.manifold.get_grain(node1_id) if hasattr(self.manifold, 'get_grain') else (self.manifold.get_node(node1_id) if hasattr(self.manifold, 'get_node') else None)
                node2 = self.manifold.get_grain(node2_id) if hasattr(self.manifold, 'get_grain') else (self.manifold.get_node(node2_id) if hasattr(self.manifold, 'get_node') else None)
                
                if node1 and node2:
                    # Create or update unresolved_tension attribute
                    if not hasattr(node1, 'unresolved_tension'):
                        node1.unresolved_tension = 0.0
                    if not hasattr(node2, 'unresolved_tension'):
                        node2.unresolved_tension = 0.0
                    
                    tension = score * 0.5
                    node1.unresolved_tension = tension
                    node2.unresolved_tension = tension
                
                # Create opposition event data
                event_data = {
                    'source_id': node1_id,
                    'target_id': node2_id,
                    'strength': score,
                    'time': self.manifold.time,
                    'is_emergent': True
                }
                
                # NEW: Add toroidal information if available
                if has_toroidal:
                    theta1, phi1 = self.manifold.get_toroidal_phase(node1_id)
                    theta2, phi2 = self.manifold.get_toroidal_phase(node2_id)
                    
                    event_data.update({
                        'source_theta': theta1,
                        'source_phi': phi1,
                        'target_theta': theta2,
                        'target_phi': phi2
                    })
                
                # Trigger opposition event
                self._trigger_event('on_opposition', event_data)
    
    def step(self):
        """
        Perform a single step of the simulation.
        
        This implements the core system evolution where field dynamics and collapse
        drive the system forward with full relational behavior.
        Enhanced with toroidal dynamics.
        """
        # Process continuous field dynamics
        self.process_continuous_fields()
        
        # Identify and select potential collapse target nodes
        collapse_targets = self.manifold.find_collapse_targets()
        
        if collapse_targets:
            # Select top target
            target = collapse_targets[0]
            
            # Find source candidates
            source_candidates = self.manifold.find_collapse_source_candidates(target['grain_id'])
            
            if source_candidates:
                # Select top source
                source = source_candidates[0]
                
                # Initiate collapse
                collapse_event = self.manifold.initiate_collapse(source['grain_id'], target['grain_id'])
                
                # Trigger collapse event
                if collapse_event:
                    # Enhanced with toroidal information
                    if self._ensure_toroidal_support():
                        # Add toroidal phase information if available
                        if 'source_phase' in collapse_event and 'target_phase' in collapse_event:
                            self._trigger_event('on_collapse', {
                                'source': source['grain_id'],
                                'target': target['grain_id'],
                                'strength': collapse_event.get('strength', 0.0),
                                'incompatible_structure': collapse_event.get('incompatible_structure', False),
                                'time': self.manifold.time,
                                'source_phase': collapse_event.get('source_phase'),
                                'target_phase': collapse_event.get('target_phase'),
                                'toroidal_phase_change': collapse_event.get('toroidal_phase_change')
                            })
                        else:
                            # Get phases directly
                            source_theta, source_phi = self.manifold.get_toroidal_phase(source['grain_id'])
                            target_theta, target_phi = self.manifold.get_toroidal_phase(target['grain_id'])
                            
                            self._trigger_event('on_collapse', {
                                'source': source['grain_id'],
                                'target': target['grain_id'],
                                'strength': collapse_event.get('strength', 0.0),
                                'incompatible_structure': collapse_event.get('incompatible_structure', False),
                                'time': self.manifold.time,
                                'source_theta': source_theta,
                                'source_phi': source_phi,
                                'target_theta': target_theta,
                                'target_phi': target_phi
                            })
                    else:
                        # Basic collapse event
                        self._trigger_event('on_collapse', {
                            'source': source['grain_id'],
                            'target': target['grain_id'],
                            'strength': collapse_event.get('strength', 0.0),
                            'incompatible_structure': collapse_event.get('incompatible_structure', False),
                            'time': self.manifold.time
                        })
        
        # Step the manifold
        self.manifold.step()
        
        # Trigger step event
        self._trigger_event('on_step', {
            'time': self.manifold.time
        })
    
    def run(self, steps: int = 1):
        """
        Run the simulation for multiple steps.
        
        Args:
            steps: Number of steps to run
        """
        for _ in range(steps):
            self.step()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the simulation.
        Enhanced with toroidal state metrics.
        
        Returns:
            Dictionary with simulation state
        """
        # Determine which collection to use (grains or nodes)
        node_collection = getattr(self.manifold, 'grains', {})
        if not node_collection and hasattr(self.manifold, 'nodes'):
            node_collection = self.manifold.nodes
            
        state = {
            'time': self.manifold.time,
            'node_count': len(node_collection),
            'activation_threshold': getattr(self.manifold, 'activation_threshold', self.config.get('activation_threshold', 0.5)),
            'config': self.config
        }
        
        # Add void-decay metrics if available
        if hasattr(self.manifold, 'get_incompatible_structure_stats'):
            void_stats = self.manifold.get_incompatible_structure_stats()
            state['void_stats'] = void_stats
        
        # NEW: Add toroidal metrics
        if self._ensure_toroidal_support():
            state['toroidal_metrics'] = self.toroidal_metrics
            
            # Add phase coherence if available
            if hasattr(self.manifold, 'calculate_phase_coherence'):
                state['phase_coherence'] = self.manifold.calculate_phase_coherence()
            
            # Add toroidal flow metrics if available
            if hasattr(self.manifold, 'calculate_toroidal_flows'):
                major_flow, minor_flow = self.manifold.calculate_toroidal_flows()
                state['major_circle_flow'] = major_flow
                state['minor_circle_flow'] = minor_flow
        
        return state