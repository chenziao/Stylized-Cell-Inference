# Neuron Parameter Inferencing

### Variables to Check
- Sampling Rate
- Conductivity of extracellular medium
    - Negative tends to be around -100 mV
    - Further away a cell gets the more it decays
    - Smallest you can detect is +- 20 mV
    

### Ideas
- Send in 3 Images as input
    - LFP Image (unscaled)
    - Everything Multiplied By 100 and cut off +- 5 mV
    - Scaled by 300 +- 1 mV