# A simpler model for AI timelines and takeoff

TL;DR a much simpler model based on the AI Futures Timelines model results in similar timelines, is more robust and more understandable

## Why make this?

- The AI Futures Model has 33 parameters; this has 8
    - AIFM is difficult to think about
- AIFM is extremely sensitive to time horizon in a way I wouldn't endorse.
    - In particular, the "doubling difficulty growth factor", which measures whether time horizon increases superexponentially, could change the date of automated coder from 2028 to 2049! I suspect that time horizon is poorly defined.

![aifm sensitivity](aifm_ac_sensitivity.png)

## Scope and limitations

- Timelines, not takeoff
- Does not model research taste

## Observations

- The AI Futures model is complex but its conclusions are fairly robust to simplifications, except in TBD ways
- The two key uncertainties behind timelines are
  - how to measure algorithmic progress
  - how effective compute relates to % automation of real tasks
- At current rates of compute growth and algorithmic progress, there will be 99% automation of AI R&D, 1e3 to 1e8 algorithmic progress, and 300x-3000x research output by 2035, even without full automation or automated research taste. This is clearly transformative AI


## More on modeling choices

### Why is automation logistic?

In other domains, ELO increases smoothly through the human range. The AIFM makes this assumption to for research taste

### Why are labor and compute Cobb-Douglas?

In the AIFM, the median estimate for substitutabilty between labor and compute is -0.15. I asked Eli why they didn't just say it was Cobb-Douglas, and he said something like Cobb-Douglas giving infinite progress if compute goes to infinity while labor remains constant, which is implausible.

- It doesn't seem so implausible to me-- it would take days to weeks to get to ASI given infinite compute, meaning a 100x-1000x speedup, but once there, infinite compute might allow developers to develop algorithms in months that would take humans billions of years with current compute levels
- Labor/compute ratio only changes by ~30x during the period in question, so it doesn't affect results much anyway
