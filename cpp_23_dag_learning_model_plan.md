# C++23 DAG Learning Model Implementation Plan

## 1. Program interface

The executable receives two required command-line arguments:

```bash
./ecodag <postfix> <row_index>
```

Arguments:

- `postfix`: string used to locate the input file `adj_mat_<postfix>.csv`.
- `row_index`: zero-based row index selecting the graph to simulate.

The program reads `adj_mat_<postfix>.csv`, extracts the selected row, and interprets it as a row-wise flattened adjacency matrix.

Example flattened adjacency matrix for a graph with 7 nodes:

```text
0111000000011100001110000111000000000000000000000
```

The number of nodes is recovered as:

```text
N = sqrt(length(flattened_matrix))
```

The program should validate that the flattened length is a perfect square. The adjacency matrix should then be reconstructed as:

```text
A[i][j] = flattened[i * N + j]
```

where `A[i][j] = 1` indicates that node `i` is a prerequisite of node `j`.

There is no special non-expressed root trait. All `N` nodes are ordinary traits, and all can have positive payoff. Agents are initialized with an empty repertoire.

---

## 2. Core objects

### 2.1 Graph

Stores:

- number of nodes `N`
- adjacency matrix
- parent masks for each node
- optional transitive ancestor masks for each node

All nodes are payoff-bearing traits. A repertoire may therefore be empty.

Useful representation:

- If `N <= 64`, use `uint64_t` bit masks for repertoires and prerequisite sets.
- If larger graphs are expected, use dynamic bitsets or sorted vectors. The first implementation can assume `N <= 64` unless we decide otherwise.

For each trait `i`, precompute:

```text
parents[i] = bit mask of direct prerequisites
ancestors[i] = bit mask of all prerequisites, if needed
```

A trait `i` is learnable for repertoire `R` if:

```text
(i not in R) and ((parents[i] & R) == parents[i])
```


---

## 3. Payoff assignments

For each graph with `N` nodes, generate payoffs for all non-root traits.

There are `N` payoff-bearing traits. Generate `N` equally spaced positive payoff values between 0 and 2 with mean 1.

Candidate baseline:

```text
payoff[k] = 2 * (k + 1) / (N + 1), for k = 0, ..., N - 1
```

This gives `N` positive values, strictly between 0 and 2, with mean 1.

Initial assignment:

```text
payoff[node_i] increases strictly with node index i
```

Then sample `P` unique permutations of the payoff vector. Each permutation is one independent payoff assignment. This ensures that payoffs are uniform in expectation with respect to graph position, while remaining variable within each focal learning environment.

Parallelization:

```text

run initialization with random asocial learners, parallelized across agents
#pragma omp parallel for
for each payoff assignment p:
    social-learning simulations / metric evaluation
```

Initialization happens only once because it is payoff-independent, so the initialized population can be reused across payoff assignments. Since geometric-age initialization samples each agent independently, both reset-rate tuning evaluations and final initialization can be parallelized across agents.

---

## 4. Population representation

Each agent has:

```text
repertoire: bit mask of learned nodes
```

Agents are initialized with an empty repertoire. No trait is automatically included.

Population-level storage:

```text
vector<uint64_t> agents;
```

or, for efficient ecology estimation:

```text
unordered_map<uint64_t, int> repertoire_counts;
```

The model needs both individual agents for direct interaction learning and repertoire counts for final ecological metrics.

---

## 5. Initialization: random asocial learner population

Goal: initialize a population using random asocial learning in the DAG. Initialization is a convenience step for creating a heterogeneous starting population, not part of the resident ecology theory.

Do not tune a reset rate for initialization. Instead, initialize each agent independently from an age-structured random-asocial learning process.

An omniscient learner has all nodes in its repertoire:

```text
R == full_mask
```

For each agent:

1. Start with an empty repertoire.
2. Draw an individual learning age `T`.
3. Let the agent perform `T` random asocial learning attempts in the DAG.

At each random asocial learning attempt:

1. Select uniformly from all currently unlearned traits.
2. If the selected trait's prerequisites are satisfied, acquire it.
3. If prerequisites are not satisfied, the attempt fails.

This means random asocial learning may target inaccessible traits, matching the later social-learning logic where attempted learning can fail.

Age distribution:

- Use a simple one-parameter age distribution.
- Preferred default: `T ~ Geometric(reset_rate)`, where `T` is the number of learning attempts since the last reset.
- Use the support `T = 0, 1, 2, ...` with `Pr(T = t) = reset_rate * (1 - reset_rate)^t`.
- Tune `reset_rate` so that approximately 5% of initialized agents are omniscient.
- Use a small grid search or bisection over `reset_rate`.

The omniscient fraction should decrease as `reset_rate` increases. This age distribution matches a constant per-step reset process: an agent's age is the number of non-reset learning opportunities since the last reset.

Outputs of initialization:

- tuned geometric reset rate
- initialized random-asocial population
- baseline repertoire distribution

---

## 6. Social-learning populations

After initialization, create two copies of the initialized population:

```text
population_payoff_biased = initialized_population
population_conformist = initialized_population
```

Each copy then evolves under its corresponding social-learning strategy for a fixed number of social-learning steps.

Strategies:

- payoff bias, strength `beta_payoff`, default `2`
- conformity, strength `beta_conformity`, default `2`

The social-learning reset rate defaults to the same reset rate tuned for the geometric initialization age distribution. This keeps the initialized age distribution aligned with the constant-turnover process used during resident social learning.

Include a fixed-reset override for sensitivity analyses.

---

## 7. Direct interaction social learning

At each social-learning step:

1. Sample one focal agent.
2. With probability `reset_rate`, reset the focal agent to the empty repertoire and stop this step.
3. Otherwise, sample `m` demonstrators from the population.
4. Each demonstrator expresses one trait sampled uniformly from its repertoire.
5. The focal agent considers expressed traits that are not already in its repertoire.
6. The focal agent assigns weights to candidate traits according to its strategy.
7. One trait is selected in proportion to weights.
8. If the selected trait is learnable given the focal repertoire, the focal acquires it. Otherwise, the attempt fails.

Optional social-learning failure-free mode:

- When `social_learning_failure_free = true`, this retry behavior applies to both payoff-biased and conformist social learning.
- If social learning selects an inaccessible trait, remove that trait from the current candidate weights and sample again from the remaining weighted candidates.
- If no weighted learnable candidate remains, the step fails.
- Random-asocial initialization retains the normal failure behavior.

Expression rule:

```text
Pr(express trait i | demonstrator repertoire R) = 1 / |R|, if i in R
```

If a demonstrator has an empty repertoire, it expresses no trait. The implementation can either skip that demonstrator or treat it as producing a null observation.

Candidate aggregation:

If multiple demonstrators express the same trait, its observed frequency/count in the sample increases.

Let `c_i` be the count of expressed candidate trait `i` among sampled demonstrators.

Payoff bias:

```text
W_i = c_i * payoff_i ^ beta_payoff
```

Conformity:

```text
W_i = c_i ^ beta_conformity
```

If no candidate trait has positive weight, the focal agent does not learn in that step.

---

## 8. Population-level visibility for final metrics

Final metrics should not be estimated from the focal learner's finite demonstrator sample. The direct interaction process generates the population, but ecological parameters are estimated from the whole resident population.

For a final population snapshot, compute trait visibility from the complete repertoire distribution.

For each trait `i`:

```text
f_i = sum over agents a of expression_probability(i | repertoire_a) / population_size
```

Under uniform expression:

```text
expression_probability(i | R_a) = 1 / |R_a|, if i in R_a and |R_a| > 0
                              = 0, otherwise
```

This gives expected population-level visibility if one random demonstrator expresses one random non-root trait from their repertoire.

---

## 9. Final metrics: eta and chi by local kappa and repertoire size

For each final population and payoff assignment, estimate ecological parameters from focal learner states.

For each unique focal repertoire `L` observed in the population, or for a sampled set of focal repertoires:

- define unlearned candidate set:

```text
U_L = all traits not in L
```

- define accessible set:

```text
A_L = traits in U_L whose prerequisites are satisfied by L
```

- define inaccessible set:

```text
I_L = U_L \ A_L
```

- define local kappa:

```text
kappa_L = |I_L| / |U_L|
```

- define repertoire size:

```text
r_L = |L|
```

For traits `i in U_L`, construct rows for a regression-like estimator:

```text
response: log visibility of trait i
predictors:
    inaccessible_i_given_L = 1 if i in I_L else 0
    centered_log_payoff_i_given_L = log(payoff_i) - mean_{j in U_L}(log(payoff_j))
```

Estimate:

```text
log f_i = alpha_bin - eta * inaccessible_i_given_L + chi * centered_log_payoff_i_given_L + error
```

where estimates are stratified by:

- resident strategy: payoff-biased or conformist
- graph-level constraint value, if available
- local kappa value
- repertoire size bin
- payoff assignment

For graphs with 8 nodes, local kappa has only a small number of possible values, so the default should store exact local kappa values rather than coarse bins.

Important: traits already in the focal repertoire are excluded from the metric calculation.

Visibility can be either raw population-level `f_i` with bin/intercept terms, or normalized within the unlearned set:

```text
f_tilde_i_given_L = f_i / sum_{j in U_L} f_j
```

Using normalized visibility is conceptually clean. Using raw visibility with focal-state or bin intercepts is mathematically close. Preferred implementation: use normalized visibility for clarity.

---

## 10. Averaging over payoff shuffles and repertoire-size stratification

The program needs to average eta and chi over payoff assignments while preserving stratification by repertoire size and local kappa.

Preferred structure:

Outer parallel loop:

```text
run conformist population once, because conformist resident dynamics is payoff-independent
for payoff_assignment in payoff_assignments:
    run payoff-biased population
    compute payoff-biased metrics using the payoff-specific resident population
    compute conformist metrics using the shared conformist resident population and current payoff assignment
    accumulate metrics into thread-local aggregators indexed by bins
```

The conformist final population is independent of payoff values because conformist learning weights depend only on observed trait counts. It should therefore be simulated once per graph and reused across payoff assignments. The conformist metrics are still evaluated for every payoff permutation, because the payoff predictor in the eta/chi regression changes.

Aggregator keys:

```text
strategy_type
kappa_bin
repertoire_size_bin
possibly graph_constraint_bin
```

Each bin stores:

```text
sum_eta
sum_chi
sum_eta_squared
sum_chi_squared
count
```

or stores sufficient statistics for pooled regression.

Important design choice:

- Averaging fitted eta/chi values from each payoff shuffle is simple and transparent.
- Pooling regression sufficient statistics across payoff shuffles is more statistically efficient but harder to inspect.

Preferred first implementation: compute eta and chi per payoff assignment and bin, then average the estimates across payoff assignments. This makes payoff-assignment variation explicit.

Sparse OLS fits should be filtered rather than reported. The default minimum is `min_ols_trait_rows = 5`, giving at least two residual degrees of freedom for an intercept plus inaccessible and payoff predictors. Additional thresholds can require a minimum number of focal repertoires per per-payoff bin and a minimum number of payoff-specific estimates before an averaged row is emitted.

Repertoire-size stratification can be implemented inside the metric calculation. It does not need to be an outer simulation loop unless we later want equal numbers of sampled focal repertoires per repertoire size.

Need to decide weighting:

1. population-weighted by observed focal repertoire frequencies;
2. equal-weighted across unique repertoires;
3. equal-weighted across repertoire-size bins.

Preferred main result: equal-weight across repertoire-size bins, with population-weighted estimates as a robustness check. This avoids letting the steady-state population overrepresent bottleneck or late-stage states.

---

## 11. Parameters

Required command-line arguments:

```text
postfix
row_index
```

Additional optional parameters with defaults:

```text
population_size = 10000
num_payoff_permutations = 5000
max_social_steps = 100 * population_size
target_omniscient_fraction = 0.05
initialization_reset_rate = tuned unless supplied
reset_rate_search_min = 0.0001
reset_rate_search_max = 0.95
reset_rate_search_iterations = 24
reset_rate_tuning_population_size = 10000
m_demonstrators = 10
beta_payoff = 2.0
beta_conformity = 2.0
social_learning_failure_free = false
fixed_reset_rate = optional social-learning override
min_ols_trait_rows = 5
min_ols_focal_repertoires = 1
min_payoff_estimates = 1
rng_seed = optional
num_threads = omp default
```

---

## 12. Outputs

For each run, write summary CSV files under `output/`.

Default output filename:

```text
output/results_avg_<postfix>_<row_index>.csv
```

The filename should include only the postfix and row index needed for later merging. Other simulation settings should appear as columns in the CSV, not in the filename.

Default output stores eta and chi averaged over payoff assignments while preserving strategy, local kappa, and repertoire-size strata. Detailed per-payoff output is optional and should only be emitted when requested, because full default runs generate many rows.

Default averaged output columns:

```text
postfix
row_index
resident_strategy
local_kappa
repertoire_size_bin
mean_eta
mean_chi
sd_eta
sd_chi
num_payoff_estimates
mean_focal_repertoires
mean_trait_focal_rows
reset_rate
beta_payoff
beta_conformity
m_demonstrators
population_size
rng_seed
social_learning_failure_free
```

Optional detailed output filename:

```text
output/results_<postfix>_<row_index>.csv
```

Optional detailed output columns:

```text
postfix
row_index
payoff_assignment_id
resident_strategy
local_kappa
repertoire_size_bin
eta
chi
num_focal_repertoires
num_trait_focal_rows
reset_rate
beta_payoff
beta_conformity
m_demonstrators
population_size
num_social_steps
rng_seed
social_learning_failure_free
```

Additional diagnostic output:

```text
initialization_reset_rate
social_reset_rate
final_omniscient_fraction
mean_repertoire_size
repertoire_size_distribution
trait_visibility_by_strategy
```

---

## 13. Resolved design decisions

1. Initialization is shared across payoff assignments.
   - Random-asocial initialization is payoff-independent, so the initialized population should be generated once per graph and reused across payoff assignments unless a later expression rule makes initialization payoff-dependent.

2. Initialization uses a tuned geometric learning-age distribution.
   - Each agent's age is drawn as `T ~ Geometric(reset_rate)`.
   - The reset rate is tuned so that approximately 5% of initialized agents are omniscient after random-asocial learning.
   - The same reset rate is then used by default for payoff-biased and conformist resident populations.
   - Reset rate is not retuned separately by strategy, because strategy-specific differences in the resulting ecology are part of the phenomenon of interest.
   - A fixed social-reset override can remain as a sensitivity option.

3. Random asocial learning selects from all unlearned traits, with failed attempts possible.
   - This mirrors the social-learning process, where agents can attempt inaccessible traits.

4. Final metric estimates use equal weighting across repertoire-size strata as the main result.
   - Population-weighted estimates should be recorded as a robustness check.
   - Repertoire-size stratification is performed during metric evaluation, not as an outer simulation loop.

5. Eta and chi are estimated with a log-linear OLS-style estimator in the first implementation.
   - This maps directly onto the reduced model and is easy to debug.
   - More elaborate likelihood-based estimators can be added later if needed.

6. For 8-node graphs, local kappa is stored as an exact value.
   - If larger graphs are later added, fixed bins across `[0,1]` can be used instead.

7. Social-learning convergence is checked using coarse windows rather than individual events.
   - Every `window_size = c * population_size` events, compute the full repertoire-frequency distribution.
   - Compare consecutive checkpoint distributions using total variation distance:

```text
D_TV(P_t, P_{t-1}) = 0.5 * sum_R |P_t(R) - P_{t-1}(R)|
```

   - Stop when the distance remains below a loose tolerance for several consecutive checks, or when `max_social_steps` is reached.
   - Candidate defaults:

```text
window_size = 5 * population_size
tolerance = 0.01 or 0.02
patience = 3 windows
```

   - For 8-node graphs, the full repertoire distribution has at most `2^8 = 256` states, so exact distribution comparison is cheap.
   - Record additional diagnostics: final omniscient fraction, mean repertoire size, trait visibility vector, and trait-visibility change across windows.
