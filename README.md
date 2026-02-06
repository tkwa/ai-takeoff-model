# A parsimonious AI timelines model predicts 99% AI R&D automation in ~2032

In this post, I describe a simple model for predicting when AI will automate AI development. It is loosely based on the [AI Futures model](https://www.timelinesmodel.com/), and makes similar predictions while being more understandable and robust. In particular, at current rates of compute growth and algorithmic progress, this model predicts >99% automation of AI R&D, 1e3 to 1e7 algorithmic progress, and 300x-3000x research output by 2035, even without full automation or automated research taste.

## Why make this?

- The AI Futures Model (AIFM) has 33 parameters; this has 8
    - The philosophy behind the AIFM is to model AI takeoff in great detail. More complex models can be more accurate, but they can also be prone to overfitting, more sensitive to modeling assumptions, and harder to understand.
- AIFM is extremely sensitive to time horizon in a way I wouldn't endorse.
    - In particular, the "doubling difficulty growth factor", which measures whether time horizon increases superexponentially, could change the date of automated coder from 2028 to 2049! I suspect that time horizon is too poorly defined to nail down this parameter.

## Scope and limitations

First, this model doesn't treat research taste and software engineering as separate skills/tasks. As such, I see it as making predictions about timelines (time to superhuman AI researcher), not takeoff (time from SAR to ASI and beyond, when increasingly superhuman research taste becomes the primary driver of progress in the AIFM). If AIs get superhuman research taste that makes AI development orders of magnitude more efficient, takeoff could be faster than this model predicts.

Second, we deliberately make two conservative assumptions:
- No full automation: as AIs get more capable, they never automate 100% of AI R&D work, just approach it.
- No substitutability: Automation follows Amdahl's law (speedup = $1/(1-f)$)

This was constructed and written up fairly quickly (about 10 hours of work), so my opinions on parameters and some of the modeling assumptions could change in the future.

## The model

We assume that AI development has the following dynamics:

- Research progress is Cobb-Douglas between labor and compute
- Software efficiency S follows a Jones model
- The fraction of automatable tasks $f$ increases as a sigmoid in log S
- Zero substitution between tasks
- Labor
    - Humans work on ONLY non-automated tasks
    - Human labor on each task is L/(1-f)
    - AI labor on each task is CS/f, but this doesn't matter because we assume human labor is the bottleneck (since humans work slower than AIs)

This implies the following model:

$$S'(t) = R(t) S^{1 - \beta} = \left(\frac L {1-f}\right)^\alpha C^\zeta S^{1 - \beta}$$

$$f(t) = \sigma(v(\log C(t)S(t) - \log E_{hac}))$$

where

- $S(t)$ is level of software (training+inference efficiency)
    - so $C(t)S(t)$ is the effective compute of the best AI
- L(t) is human labor, specified as an input time series
- C(t) is compute, also an input time series
- $\alpha, \beta, \zeta$ are constant
    - $\alpha$ is diminishing returns to more labor.
    - $\beta$ is the difficulty exponent for software improvement
    - $\zeta$ is direct returns to compute. For software intelligence explosion, this is not relevant
- $E_{hac}$ is the effective compute level of an AI that can automate half of AI R&D tasks.
- v is the automation velocity: S must increase by factor of $e^{1/v}$ to get from 50% to 73% automation

None of the components of this model are novel to the AI forecasting literature, but I haven't seen them written up in this form.

## Graphs

![Automation trajectories](plots/automation.png)
*Automation fraction f across the same 40 trajectories (logit scale). Most trajectories reach 99% automation of AI R&D by the early-to-mid 2030s.*

![40 sample trajectories](plots/trajectories.png)
*40 sampled trajectories of the model. Top left: software level S grows subexponentially (but very fast) as automation accelerates research. Top right: the parallel compute:labor ratio $C / (L/(1-f))$ (raw resource ratio before diminishing returns) decreases if automation is fast, but is ~constant if automation is on track for 99% by ~2034. Bottom left: research production R(t) increases by orders of magnitude. Bottom right: the serial compute:labor ratio $C^\zeta / (L/(1-f))^\alpha$ (with diminishing returns exponents) trends upward.*

![Sensitivity analysis](plots/sensitivity.png)
*Sensitivity analysis: median year of 99% automation as a function of each parameter, with the other parameters sampled from their prior distributions. Higher beta (diminishing returns to software improvement) and higher 1/v (slower automation) delay 99% automation the most, while the other parameters have modest effects.*

## Observations

- The AI Futures model is complex, but its conclusions are fairly robust to simplifications.
- The two key uncertainties behind timelines are
  - how to measure algorithmic progress
  - how effective compute relates to % automation of real tasks
- At current rates of compute growth and algorithmic progress, there will be 99% automation of AI R&D, 1e3 to 1e8 software efficiency gain, and 300x-3000x research output by 2035, even without full automation or automated research taste. This is clearly transformative AI
  - The median date of 99% automation is mid-2032. However, I don't put too much weight on the exact predicted timelines because I haven't thought much about the exact parameter values.
- A basic sensitivity analysis shows that higher beta (diminishing returns) and lower v (automation velocity) make 99% automation happen later, and the other parameters don't affect things much.
- Even as automation dramatically increases the amount of effective labor, the *serial* compute:labor ratio goes UP, because compute is increasing so fast and the parallel labor added by automation doesn't effectively translate into serial labor.



## Parameter values

The parameters are derived from these assumptions:

- The rate of change of S in jan 2026 is 5x/year
- 1/v is between 1.5 and 4.2
    - NB David Rein thinks 2.1 to 4.2
- f was between 0.25-0.5 in jan 2026
- alpha/(alpha + zeta) is between 0.12 and 0.35
- alpha + zeta is between 0.8 and 1
- beta is 0.3 to 1
- all variables triangularly and independently distributed
- L doubling every year until 2029 after which it increases 10%/year
- C growing 2.6x every year until 2029 after which the growth rate linearly decreases from 2x to 1.25x/year between 2030 and 2058.

For more information see the notebook: https://github.com/tkwa/ai-takeoff-model/blob/main/takeoff_simulation.ipynb

## More on modeling choices

### How could we better estimate the parameters?

We can get f_2026 [uplift fraction in 2026] from
* transcripts of realistic cursor usage + success judge + difficulty judge calibrated on tasks of known lengths
* uplift study
* asking lab employees about their current uplift (since parallel uplift and 1/(1-f) are equivalent in the simple model)

v [velocity of automation as capabilities improve] can be obtained by
* guessing the distribution of tasks, using time horizon, maybe using a correction factor for real vs benchmark time horizon
* multiple uplift studies over time
* comparing older models to newer ones, or having older models try things people use newer models for
* listing how many things get automated each year

### Why is automation logistic?

- A logistic is the simplest choice for anything that maps the reals to (0, 1).
- Intuitively, when AIs are already automating >50% of human research, each unit of capabilities progress will allow automating a constant fraction of remaining labor. The logistic has an exponential tail, which matches this intuition.

### Why are labor and compute Cobb-Douglas?

In the AIFM, the median estimate for substitutability between labor and compute is -0.15, and the plausible range includes zero (which would be Cobb-Douglas). I asked Eli why they didn't just say it was Cobb-Douglas, and he said something like Cobb-Douglas giving infinite progress if compute goes to infinity while labor remains constant, which is implausible. I have two responses to this:
- It doesn't seem so implausible to me-- it would take days to weeks to get to ASI given infinite compute, meaning a 100x-1000x speedup, but once there, infinite compute might allow developers to develop algorithms in months that would take humans billions of years with current compute levels
- Effective labor/compute ratio only changes by 10-100x during the period in question, so it doesn't affect results much anyway. The fastest trajectories are most affected by compute:labor ratio, but for trajectories that get to 99% automation around 2034, the ratio stays around 1:1.

### Why is there no substitutability between tasks?

The AIFM's median was something like $\rho = -2.0$, meaning strong complementarity. So to be conservative, I assumed no substitution effect.