## Metrics used in the Predator–Prey Analysis

I would recommend selecting two metrics per response category and using them consistently throughout the entire pipeline as the main basis for global comparison.

### 1. Predator Response Metrics

- **Predator Distance**
  - Distance between the predator and the nearest prey
  - Primarily reported normalized by the source-specific arena diagonal

- **Closing Speed**
  - Change in normalized predator-to-nearest-prey distance over a 3-step lag
  - Positive values indicate that the predator is approaching the prey
  - Evaluated as a risk-conditioned curve

- **Pursuit Alignment**
  - Alignment between the predator heading and the direction toward the nearest prey
  - Range: `[-1, 1]`
  - Evaluated as a risk-conditioned curve


### 2. Prey Response Metrics

- **Escape Alignment**
  - Alignment between prey heading and the direction away from the predator
  - Measures the instantaneous escape orientation
  - Evaluated as a risk-conditioned curve

- **Continuous Predator Response \(R\)**
  - Change in alignment toward the predator-away direction between consecutive steps
  - Measures active reorientation away from the predator
  - Evaluated as a function of predator–prey distance

- **Reaction Latency**
  - Number of steps until the first detected response after approach onset
  - Only defined for responding prey

- **Reaction Distance**
  - Predator-to-prey distance at the first detected response
  - Also available as normalized reaction distance

- **Response Fraction**
  - Fraction of classified prey responding within the response window
  - Censored prey are excluded from the denominator

- **Response Threshold Sensitivity**
  - Response fraction additionally evaluated for:
    - \(R > 0.075\)
    - \(R > 0.10\)
    - \(R > 0.125\)


### 3. Collective Response Metrics

- **Nearest-Neighbor Distance during Approach (NND)**
  - Mean prey nearest-neighbor distance
  - Reported as relative change from predator-approach onset
  - Captures contraction or expansion of the group

- **Polarization during Approach**
  - Global alignment of prey headings
  - Reported as change relative to approach onset
  - Captures directional reorganization of the group

- **Degree of Swarm (DoS)**
  - Arena-normalized mean nearest-neighbor distance
  - Whole-trajectory measure of spatial dispersion

- **Degree of Alignment (DoA)**
  - Local nearest-neighbor heading alignment
  - Whole-trajectory measure of local directional coherence

- **Cascade Size**
  - Number of prey responding within an approach event

- **Cascade Fraction**
  - Fraction of prey responding within an event
  - Group-size-normalized counterpart of cascade size

- **Propagation Time**
  - Difference between the first and last detected prey response in an event
  - Defined only for events with at least two responders

- **Mean Propagation Delay**
  - Mean delay of responding prey relative to the first responding prey



### 4. Additional Analysis

- **Group-Size Generalization**
  - Compares how expert and imitation behavior changes from 16 to 32 prey
  - Includes:
    - `delta_expert`
    - `delta_imitation`
    - `generalization_error`
    - `MAGE`

- **Imitation Fidelity**
  - Quantifies the absolute difference between expert and imitation behavior at each group size
  - Includes:
    - `fidelity_error_16`
    - `fidelity_error_32`
    - `MAFE_16`
    - `MAFE_32`

- **Heat Maps**