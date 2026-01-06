## A simpler model for AI timelines and takeoff

Suppose that AI takeoff follows the Jones model

- Cobb-Douglas between labor and compute
- Software efficiency S follows a Jones model
- Fraction of automatable tasks $f$ increases as a sigmoid in log S
- Zero substitution between tasks
- Labor
    - Humans work on ONLY non-automated tasks
    - Human labor on each task is L/(1-f)
    - AI labor on each task is CS/f, but this doesn't matter because we assume human labor bottlenecks

This implies the following model:

$$S'(t) = R(t) S^{1 - \beta} = \left(\frac L {1-f}\right)^\alpha C^\zeta S^{1 - \beta}$$

$$f(t) = \sigma(v(\log C(t)S(t) - \log E_{hac}))$$

where

- $S(t)$ is level of software (training+inference efficiency)
    - so $C(t)S(t)$ is the effective compute of the best AI
- L(t) is human labor
- C(t) is compute
- \alpha, \beta, \zeta are constant
    - $\alpha$ is diminishing returns to more labor.
    - $\beta$ is the difficulty exponent for software improvement
    - $\zeta$ is direct returns to compute. For software intelligence explosion, this is not relevant
- $E_{hac}$ is the effective compute level of the half-automated coder
- v is the automation velocity: S must increase by factor of e^(1/v) to get from 50% to 73% automation

### Differences from AI Futures model

- No full automation
- No separate coding and research test

### Parameter values

- the rate of change of S in jan 2026 is 5x/year on a log scale
- 1/v is between 1.5 and 4.2
    - NB david thinks 2.1 to 4.2
- f was between 0.25-0.5 in jan 2026
- alpha/(alpha + zeta) is between 0.15 and 0.4
- alpha + zeta is between 0.8 and 1
- beta is 0.3 to 1
- all variables triangularly and independently distributed
- L doubling every year until 2029 after which it increases 10%/year
- C tripling every year until 2029 after which the growth rate linearly decreases from 2x to 1.25x/year between 2030 and 2058.

### Implementation details for Claude etc.

If you realize it is underconstrained, tell me rather than making up other distributions

The output cells should be
- a plot with 40 trajectories of S(t) in top left panel, same trajectories of f in top right panel with a log scale (50%, 90%, 98% etc), same trajectories of research production R(t) in bottom left panel, trajectories of serial labor:serial compute ratio ($(\frac L {1-f})^\alpha / C^\zeta$) with 2026 value = 1 on bottom right panel
- Distribution of time to 99% automation
- Line plot with median time to 99% automation in each of 10 buckets conditional on a variable, with each variable on one ax
- An interactive version of the plot where one can adjust sliders
