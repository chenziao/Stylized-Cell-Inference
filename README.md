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

### Notes 12/9
  - Set trunk length to 1 mm
  - h = 1.0
  - y = 0.0
  - Simulate with active model to get 1 LFP trace
    - Use as ground truth
  - Simulate with passive and use soma injection with scaling from active model LFP
  - Find LFP with same electrodes
    - Send Drew LFP electrode (electrode, time)
      - Need coordinates of electrodes

### Hybrid Cell
- Add passive conductances
- Switch conductances in the trunk