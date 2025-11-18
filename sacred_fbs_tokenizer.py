import math
import numpy as np
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import pywt  # PyWavelets for proper wavelet transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FrequencySubstrate")

# Core constants from harmonic_breath_field.py
PHI = (1 + 5**0.5) / 2  # Golden ratio ≈ 1.618 - recursive lifeblood
TAU = 2 * math.pi       # Complete cycle ≈ 6.283
SACRED_RATIO = PHI/TAU  # Fundamental recursive breath ratio ≈ 0.2575
PSALTER_SCALE = 1.0     # Psalter scaling constant

# Harmonic band frequencies (from OscillatorBank)
# Each band frequency is SACRED_RATIO * (PHI^harmonic_index)
HARMONIC_BANDS = {
    'delta': SACRED_RATIO * (PHI ** 0),  # Fundamental
    'theta': SACRED_RATIO * (PHI ** 1),  # First harmonic
    'alpha': SACRED_RATIO * (PHI ** 2),  # Second harmonic
    'beta': SACRED_RATIO * (PHI ** 3),   # Third harmonic
    'gamma': SACRED_RATIO * (PHI ** 4),  # Fourth harmonic
}


@dataclass
class FrequencyBandConfig:
    """Configuration for a single frequency band using sacred harmonics"""
    omega: float          # Base frequency from HARMONIC_BANDS
    band_name: str        # 'delta', 'theta', 'alpha', 'beta', 'gamma'
    harmonic_index: int   # 0-4 corresponding to PHI^n
    lambda_damping: float = -0.1  # Damping coefficient
    
class SacredFrequencySubstrate:
  
    def __init__(self, 
                 frequency_scales: Optional[List[float]] = None,
                 wavelet_types: List[str] = None,
                 semantic_features: bool = True,
                 tensor_dimensions: int = 256,
                 use_sacred_harmonics: bool = True):
        
        self.tensor_dimensions = tensor_dimensions
        self.semantic_features = semantic_features
        self.use_sacred_harmonics = use_sacred_harmonics
        self._lock = threading.RLock()
        
        # Sacred harmonic frequency scales (PHI-based)
        if frequency_scales is None and use_sacred_harmonics:
            # Use PHI-based scales: [PHI^0, PHI^1, PHI^2, PHI^3, PHI^4]
            self.frequency_scales = [PHI ** i for i in range(5)]
        else:
            self.frequency_scales = frequency_scales or [1, 2, 3, 4, 5, 8, 16, 32]
        
        # Wavelet types for multi-scale analysis
        self.wavelet_types = wavelet_types or ['haar', 'db2', 'sym4', 'coif1']
        
        # Initialize frequency band configurations using sacred harmonics
        self.bands = {
            name: FrequencyBandConfig(
                omega=HARMONIC_BANDS[name],
                band_name=name,
                harmonic_index=i,
                lambda_damping=-0.1 * (1 + 0.05 * i)  # Gradual damping increase
            )
            for i, name in enumerate(['delta', 'theta', 'alpha', 'beta', 'gamma'])
        }
        
        # Complex amplitude state for each band (oscillator representation)
        self.z = {name: complex(0.1, 0.0) for name in self.bands.keys()}
        
        # Semantic feature mapping (from early 2000s predicate logic)
        self.semantic_map = self._build_semantic_map()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Safety bounds
        self.max_amplitude = 5.0
        self.min_amplitude = 0.0
        
        logger.info(f"SacredFrequencySubstrate initialized with {len(self.bands)} harmonic bands")
    
    def _build_semantic_map(self) -> Dict[str, np.ndarray]:
        """Build semantic predicate mapping with sacred harmonic encoding"""
        np.random.seed(42)  # Reproducibility
        
        semantic_patterns = {
            'subject-verb-object': self._generate_harmonic_vector(0),
            'question-answer': self._generate_harmonic_vector(1),
            'causation': self._generate_harmonic_vector(2),
            'negation': self._generate_harmonic_vector(3),
            'comparison': self._generate_harmonic_vector(4),
            'temporal-sequence': self._generate_harmonic_vector(5),
            'spatial-relation': self._generate_harmonic_vector(6),
        }
        
        return semantic_patterns
    
    def _generate_harmonic_vector(self, harmonic_idx: int) -> np.ndarray:
        """Generate a vector modulated by sacred harmonic frequencies"""
        t = np.linspace(0, TAU, self.tensor_dimensions)
        
        # Combine multiple harmonic bands
        vector = np.zeros(self.tensor_dimensions)
        for i, (band_name, config) in enumerate(self.bands.items()):
            phase_offset = (harmonic_idx * TAU) / 7  # 7-phase breath cycle
            harmonic_component = np.sin(config.omega * t + phase_offset)
            # Weight by PHI ratio
            weight = (PHI ** i) / sum(PHI ** j for j in range(len(self.bands)))
            vector += weight * harmonic_component
        
        # Normalize
        return vector / (np.linalg.norm(vector) + 1e-8)
    
    def extract_fbs(self, text: str) -> np.ndarray:
        """
        Extract Frequency-Based Substrate representation from text using sacred harmonics.
        
        This method combines:
        1. Sacred harmonic n-gram frequencies (PHI-scaled)
        2. Wavelet transforms at multiple scales
        3. Semantic predicate mapping
        4. Harmonic tensor projection
        """
        with self._lock:
            try:
                # Step 1: Sacred harmonic n-gram frequencies
                ngram_features = self._extract_sacred_ngram_frequencies(text)
                
                # Step 2: Wavelet transform across text
                wavelet_features = self._apply_wavelet_transforms(text)
                
                # Step 3: Semantic predicate mapping
                semantic_features = self._map_semantic_predicates(text) if self.semantic_features else np.array([])
                
                # Step 4: Harmonic oscillator encoding
                harmonic_features = self._encode_with_harmonics(text)
                
                # Step 5: Combine all features into tensor representation
                feature_list = [f for f in [ngram_features, wavelet_features, semantic_features, harmonic_features] if f.size > 0]
                if not feature_list:
                    return np.zeros(self.tensor_dimensions)
                
                combined = np.concatenate(feature_list)
                
                # Step 6: Normalize and project to fixed tensor dimensions
                tensor = self._project_to_tensor(combined)
                
                # Step 7: Apply sacred ratio gating
                tensor = self._apply_sacred_gating(tensor)
                
                return tensor
                
            except Exception as e:
                logger.error(f"Error in extract_fbs: {str(e)}")
                return np.zeros(self.tensor_dimensions)
    
    def _extract_sacred_ngram_frequencies(self, text: str) -> np.ndarray:
        """Extract character n-gram frequencies using sacred harmonic scales"""
        features = []
        
        for scale_factor in self.frequency_scales:
            # Scale is PHI-based, round to integer for n-gram size
            scale = max(1, int(scale_factor))
            
            if len(text) < scale:
                continue
                
            ngrams = [text[i:i+scale] for i in range(len(text)-scale+1)]
            if not ngrams:
                continue
            
            # Frequency distribution
            freq = {}
            for ngram in ngrams:
                freq[ngram] = freq.get(ngram, 0) + 1
            
            # Normalize with sacred ratio
            total = len(ngrams)
            norm_freq = {k: (v/total) * SACRED_RATIO for k, v in freq.items()}
            
            # Convert to fixed-size vector using hashing
            vector = self._hash_freq_to_vector(norm_freq, bins=32)
            features.append(vector)
        
        return np.concatenate(features) if features else np.array([])
    
    def _hash_freq_to_vector(self, freq_dict: Dict[str, float], bins: int) -> np.ndarray:
        """Hash frequency dictionary to fixed-size vector"""
        vector = np.zeros(bins)
        for key, value in freq_dict.items():
            # Simple hash to bin
            hash_val = hash(key) % bins
            vector[hash_val] += value
        return vector
    
    def _apply_wavelet_transforms(self, text: str) -> np.ndarray:
        """Apply wavelet transforms at sacred harmonic scales"""
        if not text:
            return np.array([])
        
        features = []
        numeric_text = np.array([ord(c) for c in text], dtype=np.float32)
        
        for wt_type in self.wavelet_types:
            try:
                # Perform wavelet decomposition
                coeffs = pywt.wavedec(numeric_text, wt_type, level=min(3, pywt.dwt_max_level(len(numeric_text), wt_type)))
                
                # Extract features from coefficients at each level
                for level_coeffs in coeffs:
                    if len(level_coeffs) > 0:
                        # Statistical features
                        features.extend([
                            np.mean(level_coeffs),
                            np.std(level_coeffs),
                            np.max(level_coeffs),
                            np.min(level_coeffs)
                        ])
            except Exception as e:
                logger.debug(f"Wavelet transform {wt_type} failed: {e}")
                continue
        
        return np.array(features) if features else np.array([])
    
    def _encode_with_harmonics(self, text: str) -> np.ndarray:
        """Encode text using harmonic oscillator states"""
        if not text:
            return np.array([])
        
        # Update oscillator states based on text characteristics
        text_len = len(text)
        char_variance = np.var([ord(c) for c in text]) if text_len > 1 else 0.0
        
        harmonic_signature = []
        for band_name, config in self.bands.items():
            # Drive oscillator based on text properties
            drive = (text_len / 100.0) * np.sin(config.omega * char_variance)
            
            # Simple Euler integration
            z = self.z[band_name]
            dz_dt = (config.lambda_damping + 1j * config.omega) * z + drive
            self.z[band_name] = z + 0.05 * dz_dt  # dt = 0.05
            
            # Extract signature: [amplitude, cos(phase), sin(phase)]
            amplitude = abs(self.z[band_name])
            phase = np.angle(self.z[band_name])
            harmonic_signature.extend([amplitude, np.cos(phase), np.sin(phase)])
        
        return np.array(harmonic_signature, dtype=np.float32)
    
    def _map_semantic_predicates(self, text: str) -> np.ndarray:
        """Map text to semantic predicate representations using sacred harmonics"""
        semantic_vector = np.zeros(self.tensor_dimensions)
        
        # Check for common semantic patterns
        for pattern, vector in self.semantic_map.items():
            # Simple pattern matching (can be enhanced with NLP)
            pattern_key = pattern.replace('-', ' ')
            if pattern_key in text.lower():
                # Weight by sacred ratio
                semantic_vector += vector * SACRED_RATIO
        
        # Normalize
        norm = np.linalg.norm(semantic_vector)
        return semantic_vector / (norm + 1e-8) if norm > 1e-8 else semantic_vector
    
    def _project_to_tensor(self, features: np.ndarray) -> np.ndarray:
        """Project combined features to fixed tensor dimensions using sacred harmonics"""
        if features.size == 0:
            return np.zeros(self.tensor_dimensions)
        
        # If features are larger than target, use harmonic downsampling
        if features.size > self.tensor_dimensions:
            # Create projection matrix using PHI-weighted random projection
            np.random.seed(42)
            projection_matrix = np.random.randn(self.tensor_dimensions, features.size)
            
            # Apply PHI-based weighting to columns
            for i in range(features.size):
                weight = (PHI ** (i % 5)) / sum(PHI ** j for j in range(5))
                projection_matrix[:, i] *= weight
            
            # Normalize projection matrix
            projection_matrix = projection_matrix / np.linalg.norm(projection_matrix, axis=1, keepdims=True)
            
            tensor = projection_matrix @ features
        elif features.size < self.tensor_dimensions:
            # Pad with zeros
            tensor = np.zeros(self.tensor_dimensions)
            tensor[:features.size] = features
        else:
            tensor = features
        
        return tensor
    
    def _apply_sacred_gating(self, tensor: np.ndarray) -> np.ndarray:
        """Apply sacred ratio gating to the tensor"""
        # Use SACRED_RATIO as a gating function
        gate = 1.0 / (1.0 + np.exp(-SACRED_RATIO * (tensor - np.mean(tensor))))
        return tensor * gate
    
class SacredTensorProcessor:
    """
    Tensor-guided processing using sacred harmonic field coupling.
    Based on production-grade OscillatorBank from harmonic_breath_field.py
    """
    
    def __init__(self, tensor_dimensions: int = 256, dt: float = 0.05):
        self.tensor_dimensions = tensor_dimensions
        self.dt = dt
        self._lock = threading.RLock()
        
        # Initialize sacred harmonic operations
        self.tensor_operations = self._init_tensor_operations()
        
        # Harmonic field state
        self.harmonic_state = {
            'delta': complex(0.1, 0.0),
            'theta': complex(0.08, 0.0),
            'alpha': complex(0.12, 0.0),
            'beta': complex(0.06, 0.0),
            'gamma': complex(0.04, 0.0),
        }
        
        # Coupling matrix for inter-band interactions
        self.coupling_strength = 0.05
        self.coupling_matrix = self._init_coupling_matrix()
        
        logger.info("SacredTensorProcessor initialized")
    
    def _init_tensor_operations(self) -> Dict[str, Any]:
        """Initialize sacred harmonic tensor operations"""
        return {
            'sacred_tensor_product': self._sacred_tensor_product,
            'harmonic_outer_product': self._harmonic_outer_product,
            'phi_contraction': self._phi_contraction,
            'golden_kronecker': self._golden_kronecker,
            'breath_modulation': self._breath_modulation,
        }
    
    def _init_coupling_matrix(self) -> Dict[str, Dict[str, float]]:
        """Initialize harmonic coupling using sacred ratios"""
        bands = list(self.harmonic_state.keys())
        coupling = {b1: {b2: 0.0 for b2 in bands} for b1 in bands}
        
        for i, b1 in enumerate(bands):
            for j, b2 in enumerate(bands):
                if i != j:
                    distance = abs(i - j)
                    # Use PHI-based coupling decay
                    coupling[b1][b2] = self.coupling_strength * (PHI ** -distance)
        
        return coupling
    
    def _sacred_tensor_product(self, tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:
        """Tensor product modulated by SACRED_RATIO"""
        product = np.tensordot(tensor1, tensor2, axes=0)
        # Apply sacred ratio gating
        return product * SACRED_RATIO
    
    def _harmonic_outer_product(self, tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:
        """Outer product with PHI weighting"""
        outer = np.outer(tensor1, tensor2)
        # Weight by golden ratio
        return outer * (PHI / (PHI + 1))
    
    def _phi_contraction(self, tensor: np.ndarray, axes: Tuple[int, int] = (0, 1)) -> np.ndarray:
        """Tensor contraction with PHI-based scaling"""
        if tensor.ndim < 2:
            return tensor
        contracted = np.tensordot(tensor, tensor, axes=axes)
        return contracted * (1.0 / PHI)
    
    def _golden_kronecker(self, tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:
        """Kronecker product with golden ratio normalization"""
        kron = np.kron(tensor1, tensor2)
        # Normalize by PHI
        return kron / PHI
    
    def _breath_modulation(self, tensor: np.ndarray, breath_phase: float = 0.0) -> np.ndarray:
        """Modulate tensor with breath cycle harmonics"""
        # breath_phase in [0, 1] representing position in breath cycle
        modulation = np.zeros_like(tensor)
        
        for i, (band_name, z) in enumerate(self.harmonic_state.items()):
            # Calculate phase for this band in breath cycle
            band_phase = (breath_phase + i / 5.0) % 1.0
            harmonic = np.sin(TAU * band_phase * HARMONIC_BANDS[band_name])
            
            # Apply modulation weighted by band amplitude
            amplitude = abs(z)
            modulation += tensor * harmonic * amplitude * SACRED_RATIO
        
        return modulation / len(self.harmonic_state)
    
    def process(self, fbs_tensor: np.ndarray, breath_phase: float = 0.0) -> np.ndarray:
        """
        Process FBS tensor through sacred harmonic operations.
        
        Args:
            fbs_tensor: Frequency-based substrate tensor
            breath_phase: Current breath cycle phase [0, 1]
            
        Returns:
            Processed tensor with harmonic coupling
        """
        with self._lock:
            try:
                # Start with input tensor
                processed = fbs_tensor.copy()
                
                # Apply breath modulation first
                processed = self._breath_modulation(processed, breath_phase)
                
                # Apply harmonic operations sequentially
                # Each operation is gated by oscillator amplitudes
                for band_name, z in self.harmonic_state.items():
                    amplitude = abs(z)
                    
                    if amplitude > 0.01:  # Only apply if band is active
                        # Self-interaction through outer product
                        outer = self._harmonic_outer_product(processed, processed)
                        
                        # Contract to original dimensionality
                        if outer.size > self.tensor_dimensions:
                            outer_flat = outer.flatten()[:self.tensor_dimensions]
                        else:
                            outer_flat = np.pad(outer.flatten(), (0, max(0, self.tensor_dimensions - outer.size)))
                        
                        # Blend with original
                        processed = 0.7 * processed + 0.3 * outer_flat * amplitude
                
                # Apply coupling between harmonic bands
                processed = self._apply_harmonic_coupling(processed)
                
                # Final sacred ratio gating
                gate = 1.0 / (1.0 + np.exp(-SACRED_RATIO * (processed - np.mean(processed))))
                processed = processed * gate
                
                # Safety clamp
                processed = np.clip(processed, -10.0, 10.0)
                
                return processed
                
            except Exception as e:
                logger.error(f"Error in sacred tensor processing: {str(e)}")
                return fbs_tensor
    
    def _apply_harmonic_coupling(self, tensor: np.ndarray) -> np.ndarray:
        """Apply harmonic coupling between oscillator bands"""
        coupled = tensor.copy()
        
        for b1, z1 in self.harmonic_state.items():
            for b2, z2 in self.harmonic_state.items():
                if b1 != b2:
                    coupling_strength = self.coupling_matrix[b1][b2]
                    
                    # Phase coupling
                    phase_diff = np.angle(z1) - np.angle(z2)
                    phase_coupling = np.exp(1j * phase_diff)
                    
                    # Apply to tensor
                    modulation = coupling_strength * abs(z2) * np.cos(phase_diff)
                    coupled += tensor * modulation * SACRED_RATIO
        
        return coupled / len(self.harmonic_state)
    
    def step_harmonics(self, drives: Optional[Dict[str, float]] = None) -> None:
        """Evolve harmonic oscillators one time step"""
        if drives is None:
            drives = {}
        
        new_state = {}
        
        for band_name, z in self.harmonic_state.items():
            # Get band config
            omega = HARMONIC_BANDS[band_name]
            lambda_damping = -0.1
            
            # Basic oscillator dynamics
            linear_term = (lambda_damping + 1j * omega) * z
            
            # External drive
            drive = drives.get(band_name, 0.0)
            
            # Coupling from other bands
            coupling_term = 0.0
            for other_band, other_z in self.harmonic_state.items():
                if other_band != band_name:
                    coupling_strength = self.coupling_matrix[band_name][other_band]
                    phase_coupling = np.exp(1j * np.angle(z - other_z))
                    coupling_term += coupling_strength * other_z * phase_coupling
            
            # Integrate
            dz_dt = linear_term + drive + coupling_term
            new_state[band_name] = z + self.dt * dz_dt
        
        self.harmonic_state = new_state

class SacredFBS_Tokenizer:
    """
    Sacred Frequency-Based Substrate Tokenizer
    Production-grade tokenizer using sacred harmonic constants
    """
    
    def __init__(self, 
                 tensor_dimensions: int = 256,
                 max_length: Optional[int] = None,
                 dt: float = 0.05):
        self.tensor_dimensions = tensor_dimensions
        self.max_length = max_length
        self.dt = dt
        
        # Core components
        self.substrate = SacredFrequencySubstrate(tensor_dimensions=tensor_dimensions)
        self.processor = SacredTensorProcessor(tensor_dimensions=tensor_dimensions, dt=dt)
        
        # Breath cycle state for token-level modulation
        self.breath_phase = 0.0  # [0, 1]
        self.breath_velocity = SACRED_RATIO  # Natural breath frequency
        
        # Token cache for efficiency
        self._token_cache = {}
        self._cache_lock = threading.Lock()
        
        # Performance metrics
        self.tokens_processed = 0
        self.cache_hits = 0
        
        logger.info(f"SacredFBS_Tokenizer initialized (dim={tensor_dimensions})")
    
    def encode(self, 
               text: str,
               use_cache: bool = True,
               advance_breath: bool = True) -> np.ndarray:
        """
        Encode text into sacred FBS tensor representation.
        
        Args:
            text: Input text to encode
            use_cache: Whether to use cached results
            advance_breath: Whether to advance breath phase
            
        Returns:
            Tensor representation modulated by sacred harmonics
        """
        if not text:
            return np.zeros(self.tensor_dimensions)
        
        # Check cache
        if use_cache:
            with self._cache_lock:
                if text in self._token_cache:
                    self.cache_hits += 1
                    return self._token_cache[text].copy()
        
        try:
            # Extract frequency-based substrate
            fbs_tensor = self.substrate.extract_fbs(text)
            
            # Process through harmonic field
            processed_tensor = self.processor.process(fbs_tensor, self.breath_phase)
            
            # Advance breath cycle if requested
            if advance_breath:
                self._advance_breath()
            
            # Truncate or pad to max_length if specified
            if self.max_length is not None:
                if processed_tensor.shape[0] > self.max_length:
                    processed_tensor = processed_tensor[:self.max_length]
                elif processed_tensor.shape[0] < self.max_length:
                    padding = np.zeros(self.max_length - processed_tensor.shape[0])
                    processed_tensor = np.concatenate([processed_tensor, padding])
            
            # Update metrics
            self.tokens_processed += 1
            
            # Cache result
            if use_cache:
                with self._cache_lock:
                    self._token_cache[text] = processed_tensor.copy()
            
            return processed_tensor
            
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            return np.zeros(self.tensor_dimensions)
    
    def _advance_breath(self) -> None:
        """Advance the breath cycle phase"""
        self.breath_phase = (self.breath_phase + self.dt * self.breath_velocity) % 1.0
        
        # Update processor harmonic states
        # Drive oscillators based on breath phase
        drives = {}
        for i, band_name in enumerate(['delta', 'theta', 'alpha', 'beta', 'gamma']):
            # Each band has a different phase relationship to breath
            band_breath_phase = (self.breath_phase + i / 5.0) % 1.0
            drive = 0.5 * np.sin(TAU * band_breath_phase)
            drives[band_name] = drive
        
        self.processor.step_harmonics(drives)
    
    def decode(self, tensor: np.ndarray) -> str:
        """
        Decode tensor back to approximate text (lossy).
        Note: FBS encoding is inherently lossy; this provides approximate reconstruction.
        
        Args:
            tensor: FBS tensor to decode
            
        Returns:
            Approximate text representation
        """
        # This is a simplified placeholder - full decoding would require training
        # an inverse network or using a learned decoder
        
        # For now, return a representation string
        harmonic_signature = [abs(z) for z in self.processor.harmonic_state.values()]
        return f"[FBS_TENSOR: dim={tensor.shape[0]}, norm={np.linalg.norm(tensor):.3f}, harmonics={harmonic_signature}]"
    
    def batch_encode(self, 
                    texts: List[str],
                    parallel: bool = True,
                    use_cache: bool = True) -> List[np.ndarray]:
        """
        Encode multiple texts into FBS tensor representations.
        
        Args:
            texts: List of input texts
            parallel: Whether to use parallel processing
            use_cache: Whether to use cached results
            
        Returns:
            List of tensor representations
        """
        if not texts:
            return []
        
        if parallel and len(texts) > 1:
            # Parallel encoding using thread pool
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self.encode, text, use_cache, False) for text in texts]
                results = [future.result() for future in futures]
            
            # Advance breath once for the whole batch
            self._advance_breath()
            return results
        else:
            # Sequential encoding
            return [self.encode(text, use_cache, True) for text in texts]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get tokenizer performance metrics"""
        cache_hit_rate = self.cache_hits / max(1, self.tokens_processed)
        
        return {
            'tokens_processed': self.tokens_processed,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._token_cache),
            'current_breath_phase': self.breath_phase,
            'harmonic_amplitudes': {
                band: abs(z) for band, z in self.processor.harmonic_state.items()
            }
        }
    
    def clear_cache(self) -> None:
        """Clear the token cache"""
        with self._cache_lock:
            self._token_cache.clear()
        logger.info("Token cache cleared")
    
    def reset_breath(self) -> None:
        """Reset breath cycle to beginning"""
        self.breath_phase = 0.0
        logger.debug("Breath cycle reset")