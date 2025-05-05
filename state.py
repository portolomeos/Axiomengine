"""
Enhanced version of the SimulationState class to track rotation tensor metrics,
phase continuity fields, toroidal dynamics, and Void-Decay metrics for incompatible structures.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
import time
import json
import math


class SimulationState:
    """
    Tracks and manages the state of the simulation, including metrics,
    emergent structures, and system-level properties.
    Enhanced for continuous field properties, emergent patterns, rotational dynamics,
    toroidal referencing, and Void-Decay principles.
    """
    
    def __init__(self):
        self.time = 0.0
        self.step_count = 0
        self.start_time = time.time()
        
        # System-level metrics - ADDING ALL POSSIBLE KEYS HERE
        self.metrics = {
            'total_awareness': 0.0,
            'total_collapse_metric': 0.0,
            'mean_grain_activation': 0.0,
            'mean_grain_saturation': 0.0,
            'collapse_events': 0,
            'active_nodes': 0,
            'system_entropy': 0.0,
            'system_temperature': 0.0,
            
            # Enhanced metrics for continuous fields
            'mean_field_resonance': 0.0,
            'mean_field_momentum': 0.0,
            'mean_unresolved_tension': 0.0,
            'field_coherence': 0.0,
            'continuous_flow_rate': 0.0,
            'phase_diversity': 0.0,
            
            # Rotation tensor and phase continuity metrics
            'mean_rotational_curvature': 0.0,
            'mean_phase_continuity': 0.0,
            'vortex_count': 0,
            'mean_vortex_strength': 0.0,
            'theta_mode_strength': 0.0,
            'phi_mode_strength': 0.0,
            'toroidal_coherence': 0.0,
            'dominant_theta_mode': 0,
            'dominant_phi_mode': 0,
            
            # NEW: Enhanced toroidal metrics
            'phase_coherence': 0.0,
            'major_circle_flow': 0.0,
            'minor_circle_flow': 0.0,
            'toroidal_flux': 0.0,
            'toroidal_domain_count': 0,
            'toroidal_vortex_count': 0,
            'toroidal_cluster_count': 0,
            'mean_phase_stability': 0.0,
            'cross_phase_structure_count': 0,
            
            # Void-Decay metrics for incompatible structures
            'void_region_count': 0,
            'decay_particle_count': 0,
            'mean_void_strength': 0.0,
            'mean_void_radius': 0.0,
            'incompatible_structure_rate': 0.0,
            'alignment_failure_rate': 0.0,
            'decay_emission_rate': 0.0,
            'void_affected_node_ratio': 0.0,
            'structural_tension_mean': 0.0
        }
        
        # Tracking of emergent structures
        self.structures = {
            'attractors': [],
            'confinement_zones': [],
            'recurrence_patterns': [],
            'phase_regions': {},
            
            # Enhanced structures for continuous fields
            'resonant_regions': [],
            'momentum_fields': [],
            'gradient_flows': [],
            'field_emergent_patterns': [],
            
            # Rotational structures
            'vortices': [],
            'phase_locked_regions': [],
            'toroidal_mode_rings': [],
            'theta_slices': [],
            'phi_slices': [],
            
            # NEW: Enhanced toroidal structures
            'toroidal_domains': [],
            'toroidal_vortices': [],
            'toroidal_clusters': [],
            'phase_domains': [],
            'phase_transitions': [],
            'cross_phase_structures': [],
            'phase_stability_map': {},  # Maps grain_id -> stability
            
            # Void-Decay structures
            'void_regions': [],
            'decay_particles': [],
            'incompatible_pairs': [],
            'structural_alignment_clusters': [],
            'void_clusters': []
        }
        
        # History tracking
        self.history = {
            'time': [],
            'metrics': {},
            'structures': {},
            'events': []
        }
        
        # Specialized history tracks
        self.void_formation_history = []
        self.decay_emission_history = []
        self.incompatible_structure_history = []
        self.toroidal_vortex_history = []       # NEW: Track toroidal vortices
        self.phase_transition_history = []      # NEW: Track phase transitions
        self.toroidal_resonance_history = []    # NEW: Track toroidal resonance events
        
        # Initialize history metrics
        for key in self.metrics:
            self.history['metrics'][key] = []
        
        for key in self.structures:
            if isinstance(self.structures[key], dict):
                self.history['structures'][key] = []
            else:
                self.history['structures'][key] = []
    
    def update(self, manifold):
        """
        Update the simulation state based on the current manifold state.
        Enhanced for continuous field properties, rotational dynamics,
        toroidal referencing, and Void-Decay principles.
        """
        self.time = manifold.time
        self.step_count += 1
    
        # Update system-level metrics
        total_awareness = 0.0
        total_collapse_metric = 0.0
        total_grain_activation = 0.0
        total_grain_saturation = 0.0
        active_nodes = len(manifold.grains)
    
        # Enhanced metrics for continuous fields
        total_field_resonance = 0.0
        total_field_momentum = 0.0
        total_unresolved_tension = 0.0
    
        # Rotation tensor metrics
        total_rotational_curvature = 0.0
        total_phase_continuity = 0.0
    
        # NEW: Toroidal metrics
        total_phase_stability = 0.0
    
        # Void-Decay metrics
        total_void_presence = 0.0
        nodes_with_void = 0
        total_structural_tension = 0.0
    
        # Handle case where polarity field isn't available
        polarity_field = getattr(manifold, 'epistemology_field', None)
        has_rotation_tensor = polarity_field is not None and hasattr(polarity_field, 'rotation_tensor')
    
        # NEW: Check if toroidal features are supported
        has_toroidal = hasattr(manifold, 'toroidal_phase') and hasattr(manifold, 'get_toroidal_phase')
        has_phase_stability = hasattr(manifold, 'phase_stability')
    
        # Check if Void-Decay is supported
        has_void_decay = hasattr(manifold, 'get_void_presence')
    
        for node in manifold.grains.values():
            total_awareness += node.awareness
            total_collapse_metric += node.collapse_metric
            total_grain_activation += node.grain_activation
            total_grain_saturation += node.grain_saturation
        
            # Enhanced continuous metrics
            if hasattr(node, 'field_resonance'):
                total_field_resonance += node.field_resonance
        
            if hasattr(node, 'field_momentum') and isinstance(node.field_momentum, np.ndarray) and np.any(node.field_momentum):
                total_field_momentum += np.linalg.norm(node.field_momentum)
        
            if hasattr(node, 'unresolved_tension'):
                total_unresolved_tension += node.unresolved_tension
        
            # NEW: Phase stability tracking
            if has_phase_stability and node.id in manifold.phase_stability:
                stability = manifold.phase_stability[node.id]
                total_phase_stability += stability
                # Track individual phase stability
                self.structures['phase_stability_map'][node.id] = stability
        
            # Rotational metrics (if available)
            if has_rotation_tensor:
                # Get neighbors
                related_ids = list(getattr(node, 'relations', {}).keys())
            
                # Calculate rotational curvature
                rotational_curvature = polarity_field.rotation_tensor.calculate_rotational_curvature(
                    node.id, related_ids)
                total_rotational_curvature += abs(rotational_curvature)
            
                # Get phase continuity
                phase_continuity = polarity_field.rotation_tensor.get_phase_continuity(node.id)
                total_phase_continuity += phase_continuity
        
            # Void-Decay metrics (if available)
            if has_void_decay:
                void_presence = manifold.get_void_presence(node.id)
                if void_presence > 0.05:  # Significant void presence
                    total_void_presence += void_presence
                    nodes_with_void += 1
            
                # Get structural tension if available
                config_point = manifold.config_space.get_point(node.id)
                if config_point and hasattr(config_point, 'structural_tension'):
                    total_structural_tension += config_point.structural_tension
    
        # Calculate means
        if active_nodes > 0:
            mean_grain_activation = total_grain_activation / active_nodes
            mean_grain_saturation = total_grain_saturation / active_nodes
            mean_field_resonance = total_field_resonance / active_nodes
            mean_field_momentum = total_field_momentum / active_nodes
            mean_unresolved_tension = total_unresolved_tension / active_nodes
            mean_structural_tension = total_structural_tension / active_nodes
            mean_phase_stability = total_phase_stability / active_nodes if has_phase_stability else 0.0
        
            # Rotational means
            if has_rotation_tensor:
                mean_rotational_curvature = total_rotational_curvature / active_nodes
                mean_phase_continuity = total_phase_continuity / active_nodes
            else:
                mean_rotational_curvature = 0.0
                mean_phase_continuity = 0.0
        
            # Void-Decay means
            if has_void_decay:
                mean_void_presence = total_void_presence / max(1, nodes_with_void)
                void_affected_ratio = nodes_with_void / active_nodes
            else:
                mean_void_presence = 0.0
                void_affected_ratio = 0.0
        else:
            mean_grain_activation = 0.0
            mean_grain_saturation = 0.0
            mean_field_resonance = 0.0
            mean_field_momentum = 0.0
            mean_unresolved_tension = 0.0
            mean_rotational_curvature = 0.0
            mean_phase_continuity = 0.0
            mean_void_presence = 0.0
            void_affected_ratio = 0.0
            mean_structural_tension = 0.0
            mean_phase_stability = 0.0
    
        # Update basic metrics
        self.metrics['total_awareness'] = total_awareness
        self.metrics['total_collapse_metric'] = total_collapse_metric
        self.metrics['mean_grain_activation'] = mean_grain_activation
        self.metrics['mean_grain_saturation'] = mean_grain_saturation
        self.metrics['active_nodes'] = active_nodes
    
        # Update enhanced metrics
        self.metrics['mean_field_resonance'] = mean_field_resonance
        self.metrics['mean_field_momentum'] = mean_field_momentum
        self.metrics['mean_unresolved_tension'] = mean_unresolved_tension
    
        # Update rotational metrics
        self.metrics['mean_rotational_curvature'] = mean_rotational_curvature
        self.metrics['mean_phase_continuity'] = mean_phase_continuity
    
        # NEW: Update toroidal metrics
        self.metrics['mean_phase_stability'] = mean_phase_stability
    
        # Update Void-Decay metrics
        self.metrics['structural_tension_mean'] = mean_structural_tension
        self.metrics['void_affected_node_ratio'] = void_affected_ratio
    
        # NEW: Get toroidal metrics from manifold if available
        if has_toroidal and hasattr(manifold, 'toroidal_metrics'):
            # Direct metrics from manifold's toroidal metrics
            for key, value in manifold.toroidal_metrics.items():
                if key in self.metrics:
                    self.metrics[key] = value
        
            # Calculate phase coherence if manifold provides method
            if hasattr(manifold, 'calculate_phase_coherence'):
                self.metrics['phase_coherence'] = manifold.calculate_phase_coherence()
        
            # Get toroidal flow if available
            if hasattr(manifold, 'calculate_toroidal_flows'):
                major_flow, minor_flow = manifold.calculate_toroidal_flows()
                self.metrics['major_circle_flow'] = major_flow
                self.metrics['minor_circle_flow'] = minor_flow
                # Calculate toroidal flux as magnitude of flow vector
                self.metrics['toroidal_flux'] = math.sqrt(major_flow**2 + minor_flow**2)
        
            # Get toroidal domains
            if hasattr(manifold, 'find_field_emergent_patterns'):
                patterns = manifold.find_field_emergent_patterns()
                if isinstance(patterns, dict) and 'phase_domains' in patterns:
                    self.structures['phase_domains'] = patterns['phase_domains']
                    self.metrics['toroidal_domain_count'] = len(patterns['phase_domains'])
            
                # Get cross-phase structures if available
                if 'toroidal' in patterns and 'cross_phase_structures' in patterns['toroidal']:
                    cross_structures = patterns['toroidal']['cross_phase_structures']
                    self.structures['cross_phase_structures'] = cross_structures
                    self.metrics['cross_phase_structure_count'] = len(cross_structures)
        
            # Get toroidal vortices
            if hasattr(manifold, 'detect_vortices'):
                vortices = manifold.detect_vortices()
                # Filter to keep only vortices with toroidal properties
                toroidal_vortices = [v for v in vortices if 'theta' in v and 'phi' in v]
                self.structures['toroidal_vortices'] = toroidal_vortices
                self.metrics['toroidal_vortex_count'] = len(toroidal_vortices)
        
            # Get toroidal clusters
            if hasattr(manifold, 'find_toroidal_clusters'):
                clusters = manifold.find_toroidal_clusters()
                self.structures['toroidal_clusters'] = clusters
                self.metrics['toroidal_cluster_count'] = len(clusters)
    
        # Get Void-Decay statistics from manifold if available
        if hasattr(manifold, 'get_incompatible_structure_stats'):
            void_stats = manifold.get_incompatible_structure_stats()
        
            # Update Void-Decay metrics
            self.metrics['void_region_count'] = void_stats.get('void_count', 0)
            self.metrics['decay_particle_count'] = void_stats.get('decay_count', 0)
            self.metrics['incompatible_structure_rate'] = len(void_stats.get('incompatible_pairs', [])) / (active_nodes * active_nodes) if active_nodes > 0 else 0.0
            self.metrics['alignment_failure_rate'] = void_stats.get('failure_rate', 0.0)
        
            # Update structures
            if hasattr(manifold, 'find_void_regions'):
                self.structures['void_regions'] = manifold.find_void_regions()
            
                # Calculate mean void metrics
                if self.structures['void_regions']:
                    void_strengths = [region['strength'] for region in self.structures['void_regions']]
                    void_radii = [region['radius'] for region in self.structures['void_regions']]
                
                    self.metrics['mean_void_strength'] = sum(void_strengths) / len(void_strengths) if void_strengths else 0.0
                    self.metrics['mean_void_radius'] = sum(void_radii) / len(void_radii) if void_radii else 0.0
                else:
                    self.metrics['mean_void_strength'] = 0.0
                    self.metrics['mean_void_radius'] = 0.0
        
            # Update decay emission rate
            if hasattr(manifold, 'decay_emission_events'):
                recent_emissions = [
                    event for event in manifold.decay_emission_events 
                    if event['time'] > max(0.0, self.time - 10.0)  # Events in last 10 time units
                ]
                self.metrics['decay_emission_rate'] = len(recent_emissions) / 10.0
    
        # Calculate system entropy
        # S = -∫ρ(x,t)log(ρ(x,t) + ε)dx
        entropy = 0.0
        epsilon = 1e-10  # Small constant to prevent log(0)
    
        for node in manifold.grains.values():
            if node.awareness > 0:
                entropy -= node.awareness * np.log(node.awareness + epsilon)
    
        self.metrics['system_entropy'] = entropy
    
        # Calculate system temperature (average grain activation)
        self.metrics['system_temperature'] = mean_grain_activation
    
        # Calculate field coherence (correlation between field momentum vectors)
        field_coherence = 0.0
        momentum_nodes = [node for node in manifold.grains.values() 
                        if hasattr(node, 'field_momentum') and np.any(node.field_momentum)]
    
        if len(momentum_nodes) >= 2:
            alignment_sum = 0.0
            pair_count = 0
        
            for i, node1 in enumerate(momentum_nodes):
                for node2 in momentum_nodes[i+1:]:
                    # Calculate alignment between momentum vectors
                    mom1 = node1.field_momentum
                    mom2 = node2.field_momentum
                
                    # Normalize vectors
                    norm1 = np.linalg.norm(mom1)
                    norm2 = np.linalg.norm(mom2)
                
                    if norm1 > 0 and norm2 > 0:
                        # Calculate dot product for alignment
                        dot_product = np.dot(mom1, mom2) / (norm1 * norm2)
                        # Take absolute value to measure alignment regardless of direction
                        alignment_sum += abs(dot_product)
                        pair_count += 1
        
            if pair_count > 0:
                field_coherence = alignment_sum / pair_count
    
        self.metrics['field_coherence'] = field_coherence
    
        # Update vortex metrics if rotation tensor is available
        if has_rotation_tensor:
            # Detect vortices
            vortices = polarity_field.rotation_tensor.detect_vortices(manifold.grains)
            self.structures['vortices'] = vortices
        
            # Update vortex metrics
            self.metrics['vortex_count'] = len(vortices)
        
            if vortices:
                self.metrics['mean_vortex_strength'] = sum(v['strength'] for v in vortices) / len(vortices)
            else:
                self.metrics['mean_vortex_strength'] = 0.0
        
            # Analyze toroidal structure
            toroidal_analysis = polarity_field.analyze_toroidal_structure(manifold.grains)
        
            # Store theta and phi slices
            self.structures['theta_slices'] = toroidal_analysis['theta_slices']
            self.structures['phi_slices'] = toroidal_analysis['phi_slices']
        
            # Update toroidal metrics
            self.metrics['dominant_theta_mode'] = toroidal_analysis['dominant_theta_mode']
            self.metrics['dominant_phi_mode'] = toroidal_analysis['dominant_phi_mode']
        
            # Calculate mode strengths
            theta_modes = toroidal_analysis['theta_modes']
            phi_modes = toroidal_analysis['phi_modes']
        
            if theta_modes:
                self.metrics['theta_mode_strength'] = theta_modes[0][1]
            else:
                self.metrics['theta_mode_strength'] = 0.0
            
            if phi_modes:
                self.metrics['phi_mode_strength'] = phi_modes[0][1]
            else:
                self.metrics['phi_mode_strength'] = 0.0
        
            # Calculate toroidal coherence as a combination of mode strengths
            theta_coherence = self.metrics['theta_mode_strength'] / (active_nodes + 1e-10)
            phi_coherence = self.metrics['phi_mode_strength'] / (active_nodes + 1e-10)
            self.metrics['toroidal_coherence'] = (theta_coherence + phi_coherence) / 2.0
        
            # Extract phase-locked regions
            # These are slices with high node concentrations
            phase_locked_regions = []
        
            # Check theta slices
            for i, slice_metrics in enumerate(toroidal_analysis['theta_slices']):
                if slice_metrics['node_count'] > active_nodes / len(toroidal_analysis['theta_slices']) * 1.5:
                    phase_locked_regions.append({
                        'type': 'theta',
                        'index': i,
                        'node_count': slice_metrics['node_count'],
                        'avg_awareness': slice_metrics['avg_awareness'],
                        'avg_saturation': slice_metrics['avg_saturation']
                    })
        
            # Check phi slices
            for i, slice_metrics in enumerate(toroidal_analysis['phi_slices']):
                if slice_metrics['node_count'] > active_nodes / len(toroidal_analysis['phi_slices']) * 1.5:
                    phase_locked_regions.append({
                        'type': 'phi',
                        'index': i,
                        'node_count': slice_metrics['node_count'],
                        'avg_awareness': slice_metrics['avg_awareness'],
                        'avg_saturation': slice_metrics['avg_saturation']
                    })
        
            self.structures['phase_locked_regions'] = phase_locked_regions
    
        # Get structural alignment clusters if available
        if hasattr(manifold.config_space, 'find_configuration_clusters'):
            alignment_threshold = 0.7  # Default threshold
            alignment_clusters = manifold.config_space.find_configuration_clusters(alignment_threshold)
            self.structures['structural_alignment_clusters'] = [
                {'cluster_size': len(cluster), 'nodes': list(cluster)}
                for cluster in alignment_clusters
            ]
    
        # Get void clusters if available
        if hasattr(manifold.config_space, 'find_void_clusters'):
            void_clusters = manifold.config_space.find_void_clusters()
            self.structures['void_clusters'] = [
                {'cluster_size': len(cluster), 'nodes': list(cluster)}
                for cluster in void_clusters
            ]
    
        # Record history
        self.history['time'].append(self.time)
    
        # IMPORTANT FIX: Ensure all metric keys exist in history before updating
        for key, value in self.metrics.items():
            if key not in self.history['metrics']:
                self.history['metrics'][key] = []
            self.history['metrics'][key].append(value)
    
        # Handle structures similarly - ensure keys exist before updating
        for key, value in self.structures.items():
            if key not in self.history['structures']:
                self.history['structures'][key] = []
            
            if isinstance(value, dict):
                # For dictionaries like phase regions, store count
                self.history['structures'][key].append(len(value))
            else:
                # For lists like attractors, store count
                self.history['structures'][key].append(len(value))
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event to the simulation history"""
        event = {
            'type': event_type,
            'time': self.time,
            'step': self.step_count,
            'data': data
        }
        
        self.history['events'].append(event)
        
        # Log special events to their specific histories
        if event_type == 'void_formation':
            self.void_formation_history.append(event)
        elif event_type == 'decay_emission':
            self.decay_emission_history.append(event)
        elif event_type == 'incompatible_structure':
            self.incompatible_structure_history.append(event)
        # NEW: Track toroidal-specific events
        elif event_type == 'toroidal_vortex':
            self.toroidal_vortex_history.append(event)
        elif event_type == 'phase_transition':
            self.phase_transition_history.append(event)
        elif event_type == 'toroidal_resonance':
            self.toroidal_resonance_history.append(event)
    
    def log_rotation_event(self, source_id: str, target_id: str, rotation_angle: float, phase_change: float):
        """
        Log a rotation tensor update event
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            rotation_angle: Relative rotation angle
            phase_change: Change in phase continuity
        """
        event = {
            'type': 'rotation_update',
            'time': self.time,
            'step': self.step_count,
            'data': {
                'source': source_id,
                'target': target_id,
                'rotation_angle': rotation_angle,
                'phase_change': phase_change
            }
        }
        
        self.history['events'].append(event)
    
    def log_vortex_event(self, vortex_data: Dict[str, Any]):
        """
        Log a vortex formation or change event
        
        Args:
            vortex_data: Vortex information dictionary
        """
        event = {
            'type': 'vortex_formation',
            'time': self.time,
            'step': self.step_count,
            'data': vortex_data
        }
        
        self.history['events'].append(event)
        
        # NEW: Also track toroidal vortex if toroidal data available
        if 'theta' in vortex_data and 'phi' in vortex_data:
            toroidal_event = event.copy()
            toroidal_event['type'] = 'toroidal_vortex'
            self.toroidal_vortex_history.append(toroidal_event)
    
    def log_toroidal_vortex(self, center_grain: str, strength: float, 
                           theta: float, phi: float, pattern_type: str = 'mixed',
                           rotation_direction: str = None):
        """
        NEW: Log a toroidal vortex formation or change event
        
        Args:
            center_grain: Center grain ID
            strength: Vortex strength
            theta: Theta coordinate on torus
            phi: Phi coordinate on torus
            pattern_type: Vortex pattern type ('major_circle', 'minor_circle', 'mixed')
            rotation_direction: Direction of rotation ('clockwise', 'counterclockwise')
        """
        event = {
            'type': 'toroidal_vortex',
            'time': self.time,
            'step': self.step_count,
            'data': {
                'center_grain': center_grain,
                'strength': strength,
                'theta': theta,
                'phi': phi,
                'pattern_type': pattern_type,
                'rotation_direction': rotation_direction
            }
        }
        
        self.history['events'].append(event)
        self.toroidal_vortex_history.append(event)
    
    def log_phase_transition(self, previous_coherence: float, new_coherence: float,
                            transition_type: str, affected_grains: List[str] = None):
        """
        NEW: Log a phase transition event
        
        Args:
            previous_coherence: Previous phase coherence value
            new_coherence: New phase coherence value
            transition_type: Type of transition ('ordering', 'disordering')
            affected_grains: List of affected grain IDs
        """
        event = {
            'type': 'phase_transition',
            'time': self.time,
            'step': self.step_count,
            'data': {
                'previous_coherence': previous_coherence,
                'new_coherence': new_coherence,
                'change': abs(new_coherence - previous_coherence),
                'transition_type': transition_type,
                'affected_grains': affected_grains or []
            }
        }
        
        self.history['events'].append(event)
        self.phase_transition_history.append(event)
    
    def log_toroidal_resonance(self, theta_mode: int, phi_mode: int, 
                              ratio: float, strength: float):
        """
        NEW: Log a toroidal resonance event
        
        Args:
            theta_mode: Theta mode number
            phi_mode: Phi mode number
            ratio: Ratio between modes
            strength: Strength of resonance
        """
        event = {
            'type': 'toroidal_resonance',
            'time': self.time,
            'step': self.step_count,
            'data': {
                'theta_mode': theta_mode,
                'phi_mode': phi_mode,
                'ratio': ratio,
                'strength': strength
            }
        }
        
        self.history['events'].append(event)
        self.toroidal_resonance_history.append(event)
    
    def log_void_formation(self, center_id: str, void_strength: float, affected_nodes: List[str] = None):
        """
        Log a void formation event
        
        Args:
            center_id: Center node ID of the void
            void_strength: Initial void strength
            affected_nodes: Nodes affected by the void
        """
        if affected_nodes is None:
            affected_nodes = []
        
        event = {
            'type': 'void_formation',
            'time': self.time,
            'step': self.step_count,
            'data': {
                'center_id': center_id,
                'void_strength': void_strength,
                'affected_nodes': affected_nodes
            }
        }
        
        self.history['events'].append(event)
        self.void_formation_history.append(event)
    
    def log_decay_emission(self, origin_id: str, decay_strength: float, properties: Dict[str, Any] = None):
        """
        Log a decay particle emission event
        
        Args:
            origin_id: Origin node ID of the decay
            decay_strength: Decay particle strength
            properties: Additional decay particle properties
        """
        if properties is None:
            properties = {}
        
        event = {
            'type': 'decay_emission',
            'time': self.time,
            'step': self.step_count,
            'data': {
                'origin_id': origin_id,
                'decay_strength': decay_strength,
                'properties': properties
            }
        }
        
        self.history['events'].append(event)
        self.decay_emission_history.append(event)
    
    def log_incompatible_structure(self, source_id: str, target_id: str, tension_increase: float, collapse_strength: float):
        """
        Log an incompatible structure event
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            tension_increase: Structural tension increase
            collapse_strength: Attempted collapse strength
        """
        event = {
            'type': 'incompatible_structure',
            'time': self.time,
            'step': self.step_count,
            'data': {
                'source_id': source_id,
                'target_id': target_id,
                'tension_increase': tension_increase,
                'collapse_strength': collapse_strength
            }
        }
        
        self.history['events'].append(event)
        self.incompatible_structure_history.append(event)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current simulation state"""
        summary = {
            'time': self.time,
            'step_count': self.step_count,
            'elapsed_real_time': time.time() - self.start_time,
            'metrics': self.metrics,
            'structures': {
                'attractors': len(self.structures['attractors']),
                'confinement_zones': len(self.structures['confinement_zones']),
                'recurrence_patterns': len(self.structures['recurrence_patterns']),
                'resonant_regions': len(self.structures.get('resonant_regions', [])),
                'field_emergent_patterns': len(self.structures.get('field_emergent_patterns', [])),
                'vortices': len(self.structures.get('vortices', [])),
                'phase_locked_regions': len(self.structures.get('phase_locked_regions', [])),
                'void_regions': len(self.structures.get('void_regions', [])),
                'void_clusters': len(self.structures.get('void_clusters', [])),
                'structural_alignment_clusters': len(self.structures.get('structural_alignment_clusters', []))
            }
        }
        
        # Add rotational metrics if available
        if 'mean_rotational_curvature' in self.metrics:
            summary['rotational_metrics'] = {
                'mean_rotational_curvature': self.metrics['mean_rotational_curvature'],
                'mean_phase_continuity': self.metrics['mean_phase_continuity'],
                'vortex_count': self.metrics['vortex_count'],
                'mean_vortex_strength': self.metrics['mean_vortex_strength'],
                'dominant_theta_mode': self.metrics['dominant_theta_mode'],
                'dominant_phi_mode': self.metrics['dominant_phi_mode'],
                'toroidal_coherence': self.metrics['toroidal_coherence']
            }
        
        # NEW: Add enhanced toroidal metrics
        if 'phase_coherence' in self.metrics:
            summary['toroidal_metrics'] = {
                'phase_coherence': self.metrics['phase_coherence'],
                'toroidal_domain_count': self.metrics['toroidal_domain_count'],
                'toroidal_vortex_count': self.metrics['toroidal_vortex_count'],
                'toroidal_cluster_count': self.metrics['toroidal_cluster_count'],
                'major_circle_flow': self.metrics['major_circle_flow'],
                'minor_circle_flow': self.metrics['minor_circle_flow'],
                'toroidal_flux': self.metrics['toroidal_flux'],
                'mean_phase_stability': self.metrics['mean_phase_stability'],
                'cross_phase_structure_count': self.metrics['cross_phase_structure_count']
            }
            
            # Add event counts
            summary['toroidal_metrics']['toroidal_vortex_events'] = len(self.toroidal_vortex_history)
            summary['toroidal_metrics']['phase_transition_events'] = len(self.phase_transition_history)
            summary['toroidal_metrics']['toroidal_resonance_events'] = len(self.toroidal_resonance_history)
        
        # Add Void-Decay metrics if available
        if 'void_region_count' in self.metrics:
            summary['void_decay_metrics'] = {
                'void_region_count': self.metrics['void_region_count'],
                'decay_particle_count': self.metrics['decay_particle_count'],
                'mean_void_strength': self.metrics['mean_void_strength'],
                'mean_void_radius': self.metrics['mean_void_radius'],
                'incompatible_structure_rate': self.metrics['incompatible_structure_rate'],
                'alignment_failure_rate': self.metrics['alignment_failure_rate'],
                'decay_emission_rate': self.metrics['decay_emission_rate'],
                'void_affected_node_ratio': self.metrics['void_affected_node_ratio'],
                'structural_tension_mean': self.metrics['structural_tension_mean']
            }
            
            # Add void formation and decay emission counts
            summary['void_decay_metrics']['void_formation_count'] = len(self.void_formation_history)
            summary['void_decay_metrics']['decay_emission_count'] = len(self.decay_emission_history)
            summary['void_decay_metrics']['incompatible_structure_count'] = len(self.incompatible_structure_history)
        
        return summary
    
    def export_history(self, filename: str = None, include_void_decay: bool = True, 
                      include_toroidal: bool = True) -> Dict[str, Any]:
        """
        Export the simulation history to a file or return as dictionary
        
        Args:
            filename: Output filename (if None, just returns the data)
            include_void_decay: Whether to include Void-Decay histories
            include_toroidal: Whether to include toroidal dynamics histories
            
        Returns:
            Dictionary of history data
        """
        export_data = {
            'time': self.history['time'],
            'metrics': self.history['metrics'],
            'structures': self.history['structures'],
            'events': self.history['events'][:100]  # Limit event export to prevent huge files
        }
        
        # Add Void-Decay histories if requested
        if include_void_decay:
            export_data['void_formation_history'] = self.void_formation_history
            export_data['decay_emission_history'] = self.decay_emission_history
            export_data['incompatible_structure_history'] = self.incompatible_structure_history
        
        # NEW: Add toroidal histories if requested
        if include_toroidal:
            export_data['toroidal_vortex_history'] = self.toroidal_vortex_history
            export_data['phase_transition_history'] = self.phase_transition_history
            export_data['toroidal_resonance_history'] = self.toroidal_resonance_history
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(export_data, f)
        
        return export_data
    
    def get_toroidal_analysis(self) -> Dict[str, Any]:
        """
        Get enhanced toroidal analysis data for visualization or further processing
        
        Returns:
            Dictionary with comprehensive toroidal analysis results
        """
        # Basic toroidal metrics
        toroidal_metrics = {
            'phase_coherence': self.metrics.get('phase_coherence', 0.0),
            'major_circle_flow': self.metrics.get('major_circle_flow', 0.0),
            'minor_circle_flow': self.metrics.get('minor_circle_flow', 0.0),
            'toroidal_flux': self.metrics.get('toroidal_flux', 0.0),
            'toroidal_domain_count': self.metrics.get('toroidal_domain_count', 0),
            'toroidal_vortex_count': self.metrics.get('toroidal_vortex_count', 0),
            'toroidal_cluster_count': self.metrics.get('toroidal_cluster_count', 0),
            'mean_phase_stability': self.metrics.get('mean_phase_stability', 0.0),
            'cross_phase_structure_count': self.metrics.get('cross_phase_structure_count', 0),
            'theta_mode_strength': self.metrics.get('theta_mode_strength', 0.0),
            'phi_mode_strength': self.metrics.get('phi_mode_strength', 0.0),
            'dominant_theta_mode': self.metrics.get('dominant_theta_mode', 0),
            'dominant_phi_mode': self.metrics.get('dominant_phi_mode', 0),
            'toroidal_coherence': self.metrics.get('toroidal_coherence', 0.0)
        }
        
        # Get toroidal structures
        toroidal_structures = {
            'theta_slices': self.structures.get('theta_slices', []),
            'phi_slices': self.structures.get('phi_slices', []),
            'phase_locked_regions': self.structures.get('phase_locked_regions', []),
            'toroidal_domains': self.structures.get('toroidal_domains', []),
            'toroidal_vortices': self.structures.get('toroidal_vortices', []),
            'toroidal_clusters': self.structures.get('toroidal_clusters', []),
            'phase_domains': self.structures.get('phase_domains', []),
            'cross_phase_structures': self.structures.get('cross_phase_structures', [])
        }
        
        # Get recent toroidal events
        recent_events = {
            'toroidal_vortex_events': self.toroidal_vortex_history[-min(10, len(self.toroidal_vortex_history)):],
            'phase_transition_events': self.phase_transition_history[-min(10, len(self.phase_transition_history)):],
            'toroidal_resonance_events': self.toroidal_resonance_history[-min(10, len(self.toroidal_resonance_history)):]
        }
        
        # Phase stability map (snapshot of current values)
        phase_stability = self.structures.get('phase_stability_map', {})
        
        return {
            'metrics': toroidal_metrics,
            'structures': toroidal_structures,
            'recent_events': recent_events,
            'phase_stability': phase_stability,
            'event_counts': {
                'toroidal_vortex_count': len(self.toroidal_vortex_history),
                'phase_transition_count': len(self.phase_transition_history),
                'toroidal_resonance_count': len(self.toroidal_resonance_history)
            }
        }
    
    def get_void_decay_analysis(self) -> Dict[str, Any]:
        """
        Get Void-Decay analysis data for visualization or further processing
        
        Returns:
            Dictionary with Void-Decay analysis results
        """
        # Basic void metrics
        void_metrics = {
            'void_region_count': self.metrics.get('void_region_count', 0),
            'decay_particle_count': self.metrics.get('decay_particle_count', 0),
            'mean_void_strength': self.metrics.get('mean_void_strength', 0.0),
            'mean_void_radius': self.metrics.get('mean_void_radius', 0.0),
            'void_affected_node_ratio': self.metrics.get('void_affected_node_ratio', 0.0),
            'structural_tension_mean': self.metrics.get('structural_tension_mean', 0.0),
            'incompatible_structure_rate': self.metrics.get('incompatible_structure_rate', 0.0),
            'alignment_failure_rate': self.metrics.get('alignment_failure_rate', 0.0),
            'decay_emission_rate': self.metrics.get('decay_emission_rate', 0.0)
        }
        
        # Get void and alignment structures
        void_structures = {
            'void_regions': self.structures.get('void_regions', []),
            'void_clusters': self.structures.get('void_clusters', []),
            'structural_alignment_clusters': self.structures.get('structural_alignment_clusters', [])
        }
        
        # Get recent void and decay events
        if self.void_formation_history:
            recent_void_formations = self.void_formation_history[-min(10, len(self.void_formation_history)):]
        else:
            recent_void_formations = []
            
        if self.decay_emission_history:
            recent_decay_emissions = self.decay_emission_history[-min(10, len(self.decay_emission_history)):]
        else:
            recent_decay_emissions = []
        
        return {
            'metrics': void_metrics,
            'structures': void_structures,
            'recent_void_formations': recent_void_formations,
            'recent_decay_emissions': recent_decay_emissions,
            'void_formation_count': len(self.void_formation_history),
            'decay_emission_count': len(self.decay_emission_history),
            'incompatible_structure_count': len(self.incompatible_structure_history)
        }
    
    def visualize_toroidal_metrics(self):
        """
        Create a visualization of toroidal metrics using matplotlib
        (placeholder - actual visualization would be implemented in a separate module)
        """
        # Placeholder for visualization functionality
        # This would be better implemented in a visualization module
        print("Toroidal Metrics Visualization:")
        print(f"Phase coherence: {self.metrics.get('phase_coherence', 0.0):.4f}")
        print(f"Major circle flow: {self.metrics.get('major_circle_flow', 0.0):.4f}")
        print(f"Minor circle flow: {self.metrics.get('minor_circle_flow', 0.0):.4f}")
        print(f"Toroidal flux: {self.metrics.get('toroidal_flux', 0.0):.4f}")
        print(f"Toroidal domain count: {self.metrics.get('toroidal_domain_count', 0)}")
        print(f"Toroidal vortex count: {self.metrics.get('toroidal_vortex_count', 0)}")
        print(f"Mean phase stability: {self.metrics.get('mean_phase_stability', 0.0):.4f}")
        print(f"Dominant theta mode: {self.metrics.get('dominant_theta_mode', 0)}")
        print(f"Dominant phi mode: {self.metrics.get('dominant_phi_mode', 0)}")
    
    def visualize_void_decay_metrics(self):
        """
        Create a visualization of Void-Decay metrics using matplotlib
        (placeholder - actual visualization would be implemented in a separate module)
        """
        # Placeholder for visualization functionality
        # This would be better implemented in a visualization module
        print("Void-Decay Metrics Visualization:")
        print(f"Void region count: {self.metrics.get('void_region_count', 0)}")
        print(f"Decay particle count: {self.metrics.get('decay_particle_count', 0)}")
        print(f"Mean void strength: {self.metrics.get('mean_void_strength', 0.0):.4f}")
        print(f"Mean void radius: {self.metrics.get('mean_void_radius', 0.0):.4f}")
        print(f"Incompatible structure rate: {self.metrics.get('incompatible_structure_rate', 0.0):.4f}")
        print(f"Alignment failure rate: {self.metrics.get('alignment_failure_rate', 0.0):.4f}")
        print(f"Void affected node ratio: {self.metrics.get('void_affected_node_ratio', 0.0):.4f}")


# Factory function for easier creation
def create_simulation_state() -> SimulationState:
    """Create a new simulation state instance"""
    return SimulationState()