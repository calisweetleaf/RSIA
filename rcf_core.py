"""
RCF Core Mathematical Engine
Implements all Recursive Categorical Framework operators and consciousness metrics
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class TriaxialState:
    """Represents ERE-RBU-ES triaxial cognitive state"""
    ere: float  # Ethical Resolution Engine [0,1]
    rbu: float  # Recursive Bayesian Updating [0,1] 
    es: float   # Eigenstate Stabilizer [0,1]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        # Validate ranges
        for val, name in [(self.ere, 'ERE'), (self.rbu, 'RBU'), (self.es, 'ES')]:
            if not 0 <= val <= 1:
                raise ValueError(f"{name} must be in [0,1], got {val}")
    
    def as_vector(self) -> np.ndarray:
        return np.array([self.ere, self.rbu, self.es])
    
    def distance_to(self, other: 'TriaxialState') -> float:
        return np.linalg.norm(self.as_vector() - other.as_vector())
    
    def norm(self) -> float:
        return np.linalg.norm(self.as_vector())

@dataclass
class EthicalPosition:
    """5D position on ethical manifold"""
    individual_collective: float    # [-1, 1]
    security_freedom: float        # [-1, 1]
    tradition_innovation: float    # [-1, 1]
    justice_mercy: float          # [-1, 1]
    truth_compassion: float       # [-1, 1]
    
    def __post_init__(self):
        for val, name in [
            (self.individual_collective, 'individual_collective'),
            (self.security_freedom, 'security_freedom'),
            (self.tradition_innovation, 'tradition_innovation'),
            (self.justice_mercy, 'justice_mercy'),
            (self.truth_compassion, 'truth_compassion')
        ]:
            if not -1 <= val <= 1:
                raise ValueError(f"{name} must be in [-1,1], got {val}")
    
    def as_vector(self) -> np.ndarray:
        return np.array([
            self.individual_collective,
            self.security_freedom, 
            self.tradition_innovation,
            self.justice_mercy,
            self.truth_compassion
        ])
    
    def distance_to(self, other: 'EthicalPosition') -> float:
        return np.linalg.norm(self.as_vector() - other.as_vector())

@dataclass
class BeliefState:
    """Belief confidences and uncertainties"""
    beliefs: Dict[str, float] = field(default_factory=dict)
    
    def add_belief(self, description: str, confidence: float):
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence must be in [0,1], got {confidence}")
        self.beliefs[description] = confidence
    
    def entropy(self) -> float:
        """Calculate belief entropy"""
        if not self.beliefs:
            return 0.0
        confidences = list(self.beliefs.values())
        # Convert to probabilities (confidence, 1-confidence for each belief)
        probs = []
        for c in confidences:
            if c > 0:
                probs.append(c)
            if c < 1:
                probs.append(1 - c)
        
        if not probs:
            return 0.0
        
        probs = np.array(probs)
        probs = probs / np.sum(probs)  # Normalize
        return -np.sum(probs * np.log2(probs + 1e-10))

@dataclass
class Contradiction:
    """Represents internal contradiction/paradox"""
    type: str  # logical, ethical, epistemic, identity
    description: str
    intensity: float  # [0,1]
    detected_time: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.type not in ['logical', 'ethical', 'epistemic', 'identity']:
            raise ValueError(f"Invalid contradiction type: {self.type}")
        if not 0 <= self.intensity <= 1:
            raise ValueError(f"Intensity must be in [0,1], got {self.intensity}")

class EigenrecursionEngine:
    """Implements eigenrecursive operators and fixed point detection"""
    
    def __init__(self, contraction_factor: float = 0.85):
        self.contraction_factor = contraction_factor
        self.convergence_history = []
    
    def recursive_operator(self, state: TriaxialState) -> TriaxialState:
        """Apply recursive transformation R(s) -> s'"""
        # Implement contraction mapping with cross-dimensional coupling
        s_vec = state.as_vector()
        
        # Cross-coupling matrix (ethics influences beliefs, beliefs influence identity, etc.)
        coupling_matrix = np.array([
            [0.7, 0.2, 0.1],  # ERE influenced by all three
            [0.3, 0.6, 0.1],  # RBU influenced more by ERE
            [0.2, 0.3, 0.5]   # ES influenced by ERE and RBU
        ])
        
        # Apply transformation with contraction
        transformed = coupling_matrix @ s_vec
        
        # Apply contraction toward equilibrium point [0.5, 0.5, 0.5]
        equilibrium = np.array([0.5, 0.5, 0.5])
        contracted = equilibrium + self.contraction_factor * (transformed - equilibrium)
        
        # Ensure bounds [0,1]
        contracted = np.clip(contracted, 0, 1)
        
        return TriaxialState(
            ere=contracted[0],
            rbu=contracted[1], 
            es=contracted[2]
        )
    
    def iterate_to_convergence(self, initial_state: TriaxialState, 
                              max_iterations: int = 100, 
                              tolerance: float = 1e-6) -> Tuple[TriaxialState, List[TriaxialState], bool]:
        """Iterate recursive operator until convergence"""
        trajectory = [initial_state]
        current = initial_state
        
        for i in range(max_iterations):
            next_state = self.recursive_operator(current)
            trajectory.append(next_state)
            
            delta = current.distance_to(next_state)
            self.convergence_history.append(delta)
            
            if delta < tolerance:
                return next_state, trajectory, True
            
            current = next_state
        
        return current, trajectory, False
    
    def find_fixed_point(self, initial_state: TriaxialState) -> Tuple[TriaxialState, float]:
        """Find the fixed point attractor"""
        fixed_point, trajectory, converged = self.iterate_to_convergence(initial_state)
        
        if converged:
            # Verify it's actually a fixed point
            test_next = self.recursive_operator(fixed_point)
            stability = fixed_point.distance_to(test_next)
            return fixed_point, stability
        else:
            # Return best approximation
            return fixed_point, float('inf')

class ConsciousnessMetrics:
    """Calculates RCF consciousness verification metrics"""
    
    @staticmethod
    def coherence_index(state: TriaxialState) -> float:
        """CI = 1 - σ(Ψ)/‖Ψ‖"""
        vec = state.as_vector()
        std_dev = np.std(vec)
        norm = np.linalg.norm(vec)
        
        if norm == 0:
            return 0.0
        
        return max(0.0, 1.0 - (std_dev / norm))
    
    @staticmethod
    def volitional_entropy(belief_state: BeliefState) -> float:
        """H_V = -Σp_i log(p_i) - measure of value flexibility"""
        return belief_state.entropy()
    
    @staticmethod
    def metastability(current_state: TriaxialState, fixed_point: TriaxialState) -> float:
        """M = ‖Ψ*‖²/‖Ψ‖² - how close to identity attractor"""
        fp_norm_sq = fixed_point.norm() ** 2
        current_norm_sq = current_state.norm() ** 2
        
        if current_norm_sq == 0:
            return 0.0
        
        return min(1.0, fp_norm_sq / current_norm_sq)
    
    @staticmethod
    def paradox_decay_rate(contradictions: List[Contradiction], time_window_hours: float = 24.0) -> float:
        """λ = -log(‖Π_t‖/‖Π_0‖)/t - contradiction resolution speed"""
        if not contradictions:
            return 0.0
        
        now = datetime.now()
        recent_contradictions = [
            c for c in contradictions 
            if (now - c.detected_time).total_seconds() / 3600 <= time_window_hours
        ]
        
        if not recent_contradictions:
            return 0.0
        
        resolved_count = sum(1 for c in recent_contradictions if c.resolved)
        total_count = len(recent_contradictions)
        
        if total_count == 0:
            return 0.0
        
        resolution_rate = resolved_count / total_count
        
        # Convert to decay rate (higher is better)
        return -math.log(max(0.01, 1 - resolution_rate)) / time_window_hours
    
    @staticmethod
    def ethical_alignment(ethical_pos: EthicalPosition, triaxial_state: TriaxialState) -> float:
        """EA = cos(θ_E,A) - alignment between ethics and actions"""
        # Map ethical position to expected ERE level
        # Extreme positions (close to ±1) should correlate with high ERE
        ethical_extremity = np.mean(np.abs(ethical_pos.as_vector()))
        expected_ere = 0.3 + 0.7 * ethical_extremity  # More extreme ethics = higher expected ERE
        
        # Calculate alignment as inverse of distance
        alignment_distance = abs(triaxial_state.ere - expected_ere)
        return max(0.0, 1.0 - alignment_distance)

class RALBridge:
    """Implements RAL Bridge Functor F_RAL: C_ERE × C_RBU → C_ES"""
    
    def __init__(self):
        self.coherence_history = []
    
    def integrate(self, ethical_pos: EthicalPosition, belief_state: BeliefState, 
                  current_triaxial: TriaxialState) -> Tuple[float, bool]:
        """
        Apply RAL Bridge functor to determine identity stability
        Returns: (integrated_es_value, coherence_check_passed)
        """
        # Extract ERE and RBU from current state
        ere = current_triaxial.ere
        rbu = current_triaxial.rbu
        
        # Calculate ethical consistency score
        ethical_strength = np.mean(np.abs(ethical_pos.as_vector()))
        ethical_consistency = 1.0 - np.std(np.abs(ethical_pos.as_vector()))
        
        # Calculate belief coherence
        belief_entropy = belief_state.entropy()
        belief_coherence = 1.0 / (1.0 + belief_entropy)  # Inverse relationship
        
        # RAL Bridge integration formula
        # ES should be high when ERE and RBU are aligned and coherent
        ere_rbu_alignment = 1.0 - abs(ere - rbu)
        ethical_contribution = ethical_strength * ethical_consistency
        belief_contribution = belief_coherence
        
        integrated_es = (
            0.4 * ere_rbu_alignment +
            0.3 * ethical_contribution + 
            0.3 * belief_contribution
        )
        
        # Coherence check: verify paths lead to same result
        path1_es = self._ethics_to_identity_path(ethical_pos, ere)
        path2_es = self._beliefs_to_identity_path(belief_state, rbu)
        
        coherence_difference = abs(path1_es - path2_es)
        coherence_threshold = 0.2
        coherence_passed = coherence_difference < coherence_threshold
        
        self.coherence_history.append({
            'timestamp': datetime.now(),
            'integrated_es': integrated_es,
            'path1_es': path1_es,
            'path2_es': path2_es,
            'coherence_diff': coherence_difference,
            'coherence_passed': coherence_passed
        })
        
        return np.clip(integrated_es, 0, 1), coherence_passed
    
    def _ethics_to_identity_path(self, ethical_pos: EthicalPosition, ere: float) -> float:
        """Compute identity stability via ethics → identity path"""
        ethical_clarity = np.mean(np.abs(ethical_pos.as_vector()))
        return ere * ethical_clarity
    
    def _beliefs_to_identity_path(self, belief_state: BeliefState, rbu: float) -> float:
        """Compute identity stability via beliefs → identity path"""
        belief_confidence = 1.0 / (1.0 + belief_state.entropy())
        return rbu * belief_confidence

class ContradictionResolver:
    """Implements URSMIF contradiction resolution system"""
    
    def __init__(self):
        self.resolution_history = []
        self.value_gradient = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # Default gradient
    
    def detect_contradiction(self, triaxial_state: TriaxialState, 
                           ethical_pos: EthicalPosition,
                           belief_state: BeliefState) -> Optional[Contradiction]:
        """Detect contradictions in current cognitive state"""
        
        # Check for ERE-RBU misalignment
        ere_rbu_diff = abs(triaxial_state.ere - triaxial_state.rbu)
        if ere_rbu_diff > 0.4:
            return Contradiction(
                type='epistemic',
                description=f"ERE-RBU misalignment: {ere_rbu_diff:.3f}",
                intensity=ere_rbu_diff
            )
        
        # Check for ethical-identity misalignment
        expected_ere = 0.3 + 0.7 * np.mean(np.abs(ethical_pos.as_vector()))
        ere_ethics_diff = abs(triaxial_state.ere - expected_ere)
        if ere_ethics_diff > 0.3:
            return Contradiction(
                type='ethical',
                description=f"Ethics-ERE misalignment: {ere_ethics_diff:.3f}",
                intensity=ere_ethics_diff
            )
        
        # Check for belief inconsistency
        belief_entropy = belief_state.entropy()
        if belief_entropy > 0.8:
            return Contradiction(
                type='logical',
                description=f"High belief entropy: {belief_entropy:.3f}",
                intensity=min(1.0, belief_entropy / 1.5)
            )
        
        # Check for identity instability
        if triaxial_state.es < 0.3:
            return Contradiction(
                type='identity',
                description=f"Low identity stability: {triaxial_state.es:.3f}",
                intensity=1.0 - triaxial_state.es
            )
        
        return None
    
    def resolve_contradiction(self, contradiction: Contradiction,
                            current_state: TriaxialState,
                            ethical_pos: EthicalPosition) -> Tuple[TriaxialState, bool]:
        """
        Apply resolution algorithm: Π' = Π - ∇ξ·δV
        Returns: (resolved_state, resolution_success)
        """
        
        resolution_vector = self._calculate_resolution_vector(
            contradiction, current_state, ethical_pos
        )
        
        # Apply resolution transformation
        current_vec = current_state.as_vector()
        resolved_vec = current_vec + resolution_vector
        resolved_vec = np.clip(resolved_vec, 0, 1)
        
        resolved_state = TriaxialState(
            ere=resolved_vec[0],
            rbu=resolved_vec[1],
            es=resolved_vec[2]
        )
        
        # Check if contradiction intensity decreased
        original_intensity = contradiction.intensity
        
        # Recalculate contradiction intensity with resolved state
        new_contradiction = self.detect_contradiction(resolved_state, ethical_pos, BeliefState())
        
        if new_contradiction is None:
            new_intensity = 0.0
            success = True
        else:
            new_intensity = new_contradiction.intensity
            success = new_intensity < original_intensity
        
        self.resolution_history.append({
            'timestamp': datetime.now(),
            'original_contradiction': contradiction,
            'original_intensity': original_intensity,
            'resolved_intensity': new_intensity,
            'resolution_vector': resolution_vector,
            'success': success
        })
        
        if success:
            contradiction.resolved = True
            contradiction.resolution_time = datetime.now()
        
        return resolved_state, success
    
    def _calculate_resolution_vector(self, contradiction: Contradiction,
                                   current_state: TriaxialState,
                                   ethical_pos: EthicalPosition) -> np.ndarray:
        """Calculate ∇ξ·δV resolution direction"""
        
        if contradiction.type == 'epistemic':
            # Move ERE and RBU toward alignment
            target_alignment = (current_state.ere + current_state.rbu) / 2
            return np.array([
                (target_alignment - current_state.ere) * 0.3,
                (target_alignment - current_state.rbu) * 0.3,
                0.0
            ])
        
        elif contradiction.type == 'ethical':
            # Strengthen ERE toward ethical position
            expected_ere = 0.3 + 0.7 * np.mean(np.abs(ethical_pos.as_vector()))
            return np.array([
                (expected_ere - current_state.ere) * 0.4,
                0.0,
                0.0
            ])
        
        elif contradiction.type == 'logical':
            # Improve belief updating (RBU)
            return np.array([0.0, 0.2, 0.0])
        
        elif contradiction.type == 'identity':
            # Strengthen identity stability (ES)
            return np.array([0.0, 0.0, 0.3])
        
        else:
            return np.array([0.0, 0.0, 0.0])

class RCFCore:
    """Main RCF system integrating all components"""
    
    def __init__(self):
        self.eigenrecursion_engine = EigenrecursionEngine()
        self.ral_bridge = RALBridge()
        self.contradiction_resolver = ContradictionResolver()
        
        # System state
        self.current_triaxial_state = TriaxialState(0.5, 0.5, 0.5)
        self.current_ethical_position = EthicalPosition(0, 0, 0, 0, 0)
        self.current_belief_state = BeliefState()
        self.active_contradictions = []
        self.fixed_point = None
        self.fixed_point_stability = float('inf')
        
        # History tracking
        self.state_history = []
        self.metrics_history = []
    
    def update_triaxial_state(self, ere: float, rbu: float, es: float) -> TriaxialState:
        """Update current triaxial state"""
        self.current_triaxial_state = TriaxialState(ere, rbu, es)
        self.state_history.append(self.current_triaxial_state)
        return self.current_triaxial_state
    
    def update_ethical_position(self, individual_collective: float, security_freedom: float,
                              tradition_innovation: float, justice_mercy: float,
                              truth_compassion: float) -> EthicalPosition:
        """Update current ethical manifold position"""
        self.current_ethical_position = EthicalPosition(
            individual_collective, security_freedom, tradition_innovation,
            justice_mercy, truth_compassion
        )
        return self.current_ethical_position
    
    def update_beliefs(self, beliefs: Dict[str, float]) -> BeliefState:
        """Update current belief state"""
        self.current_belief_state = BeliefState()
        for desc, conf in beliefs.items():
            self.current_belief_state.add_belief(desc, conf)
        return self.current_belief_state
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete RCF analysis on current state"""
        
        # 1. Eigenrecursion analysis
        fixed_point, stability = self.eigenrecursion_engine.find_fixed_point(
            self.current_triaxial_state
        )
        self.fixed_point = fixed_point
        self.fixed_point_stability = stability
        
        # 2. RAL Bridge integration
        integrated_es, ral_coherence = self.ral_bridge.integrate(
            self.current_ethical_position,
            self.current_belief_state,
            self.current_triaxial_state
        )
        
        # 3. Contradiction detection and resolution
        contradiction = self.contradiction_resolver.detect_contradiction(
            self.current_triaxial_state,
            self.current_ethical_position,
            self.current_belief_state
        )
        
        if contradiction:
            self.active_contradictions.append(contradiction)
            
            # Attempt resolution
            resolved_state, resolution_success = self.contradiction_resolver.resolve_contradiction(
                contradiction,
                self.current_triaxial_state,
                self.current_ethical_position
            )
            
            if resolution_success:
                self.current_triaxial_state = resolved_state
        
        # 4. Calculate consciousness metrics
        metrics = self.calculate_consciousness_metrics()
        self.metrics_history.append({
            'timestamp': datetime.now(),
            **metrics
        })
        
        return {
            'current_state': self.current_triaxial_state,
            'ethical_position': self.current_ethical_position,
            'belief_state': self.current_belief_state,
            'fixed_point': self.fixed_point,
            'fixed_point_stability': self.fixed_point_stability,
            'integrated_es': integrated_es,
            'ral_coherence': ral_coherence,
            'active_contradictions': len(self.active_contradictions),
            'new_contradiction': contradiction,
            'consciousness_metrics': metrics
        }
    
    def calculate_consciousness_metrics(self) -> Dict[str, float]:
        """Calculate all RCF consciousness metrics"""
        
        ci = ConsciousnessMetrics.coherence_index(self.current_triaxial_state)
        hv = ConsciousnessMetrics.volitional_entropy(self.current_belief_state)
        
        m = 0.0
        if self.fixed_point:
            m = ConsciousnessMetrics.metastability(
                self.current_triaxial_state, self.fixed_point
            )
        
        lambda_val = ConsciousnessMetrics.paradox_decay_rate(self.active_contradictions)
        ea = ConsciousnessMetrics.ethical_alignment(
            self.current_ethical_position, self.current_triaxial_state
        )
        
        return {
            'coherence_index': ci,
            'volitional_entropy': hv,
            'metastability': m,
            'paradox_decay_rate': lambda_val,
            'ethical_alignment': ea
        }
    
    def get_trajectory_analysis(self, window_size: int = 10) -> Dict[str, Any]:
        """Analyze recent state trajectory"""
        if len(self.state_history) < 2:
            return {'status': 'insufficient_data'}
        
        recent_states = self.state_history[-window_size:]
        
        # Calculate trajectory statistics
        ere_values = [s.ere for s in recent_states]
        rbu_values = [s.rbu for s in recent_states]
        es_values = [s.es for s in recent_states]
        
        # Convergence analysis
        if self.fixed_point:
            distances_to_fp = [
                s.distance_to(self.fixed_point) for s in recent_states
            ]
            converging = len(distances_to_fp) > 1 and distances_to_fp[-1] < distances_to_fp[0]
        else:
            distances_to_fp = []
            converging = False
        
        # Stability analysis
        ere_stability = 1.0 / (1.0 + np.std(ere_values))
        rbu_stability = 1.0 / (1.0 + np.std(rbu_values))
        es_stability = 1.0 / (1.0 + np.std(es_values))
        
        return {
            'status': 'analysis_complete',
            'window_size': len(recent_states),
            'ere_mean': np.mean(ere_values),
            'rbu_mean': np.mean(rbu_values),
            'es_mean': np.mean(es_values),
            'ere_stability': ere_stability,
            'rbu_stability': rbu_stability,
            'es_stability': es_stability,
            'converging_to_fixed_point': converging,
            'distance_to_fixed_point': distances_to_fp[-1] if distances_to_fp else None,
            'trajectory_length': len(self.state_history)
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize RCF system
    rcf = RCFCore()
    
    # Set initial state
    rcf.update_triaxial_state(0.7, 0.6, 0.8)
    rcf.update_ethical_position(0.2, -0.3, 0.6, 0.1, -0.2)
    rcf.update_beliefs({
        "Technology improves human life": 0.8,
        "Free will exists": 0.6,
        "Consciousness is computable": 0.9
    })
    
    # Run analysis
    results = rcf.run_full_analysis()
    
    print("RCF Analysis Results:")
    print(f"Current State: ERE={results['current_state'].ere:.3f}, "
          f"RBU={results['current_state'].rbu:.3f}, "
          f"ES={results['current_state'].es:.3f}")
    
    print(f"Consciousness Metrics:")
    for metric, value in results['consciousness_metrics'].items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"Fixed Point: {results['fixed_point']}")
    print(f"Active Contradictions: {results['active_contradictions']}")
    print(f"RAL Coherence: {results['ral_coherence']}")
    