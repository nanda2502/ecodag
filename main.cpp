#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using Mask = std::uint64_t;

struct Options {
    std::string postfix;
    std::size_t row_index = 0;
    int population_size = 10'000;
    int num_payoff_permutations = 5'000;
    int max_social_steps = -1;
    double target_omniscient_fraction = 0.05;
    std::optional<double> initialization_reset_rate;
    double reset_rate_search_min = 1e-4;
    double reset_rate_search_max = 0.95;
    int reset_rate_search_iterations = 24;
    int reset_rate_tuning_population_size = 10'000;
    int m_demonstrators = 10;
    double beta_payoff = 2.0;
    double beta_conformity = 2.0;
    std::optional<double> fixed_reset_rate;
    std::uint64_t rng_seed = 1;
    int num_threads = 0;
    int convergence_window_multiplier = 5;
    double convergence_tolerance = 0.01;
    int convergence_patience = 3;
    int min_ols_trait_rows = 5;
    int min_ols_focal_repertoires = 1;
    int min_payoff_estimates = 1;
    bool social_learning_failure_free = false;
    bool emit_detailed_results = false;
};

struct Graph {
    int n = 0;
    std::vector<Mask> parents;
    Mask full_mask = 0;

    [[nodiscard]] bool learnable(Mask repertoire, int trait) const {
        const Mask bit = Mask{1} << trait;
        return (repertoire & bit) == 0 && (parents[trait] & repertoire) == parents[trait];
    }
};

struct PopulationSummary {
    double omniscient_fraction = 0.0;
    double mean_repertoire_size = 0.0;
};

struct SimulationDiagnostics {
    int steps_run = 0;
    double final_tv_distance = std::numeric_limits<double>::quiet_NaN();
};

struct MetricRow {
    std::string strategy;
    int payoff_assignment_id = 0;
    double local_kappa = 0.0;
    int repertoire_size_bin = 0;
    double eta = std::numeric_limits<double>::quiet_NaN();
    double chi = std::numeric_limits<double>::quiet_NaN();
    int num_focal_repertoires = 0;
    int num_trait_focal_rows = 0;
    double final_omniscient_fraction = 0.0;
    double mean_repertoire_size = 0.0;
    int steps_run = 0;
    double final_tv_distance = std::numeric_limits<double>::quiet_NaN();
};

struct AveragedMetricRow {
    std::string strategy;
    double local_kappa = 0.0;
    int repertoire_size_bin = 0;
    double mean_eta = std::numeric_limits<double>::quiet_NaN();
    double mean_chi = std::numeric_limits<double>::quiet_NaN();
    double sd_eta = std::numeric_limits<double>::quiet_NaN();
    double sd_chi = std::numeric_limits<double>::quiet_NaN();
    int num_payoff_estimates = 0;
    double mean_focal_repertoires = 0.0;
    double mean_trait_focal_rows = 0.0;
};

enum class Strategy {
    PayoffBiased,
    Conformist,
};

[[nodiscard]] std::uint64_t parse_u64(const std::string& text) {
    return static_cast<std::uint64_t>(std::stoull(text));
}

[[nodiscard]] int parse_int(const std::string& text) {
    return std::stoi(text);
}

[[nodiscard]] double parse_double(const std::string& text) {
    return std::stod(text);
}

[[nodiscard]] std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

[[nodiscard]] Options parse_args(int argc, char** argv) {
    if (argc < 3) {
        throw std::runtime_error(
            "Usage: ecodag <postfix> <row_index> [--population-size N] "
            "[--num-payoff-permutations N] [--max-social-steps N] [--seed N]");
    }

    Options options;
    options.postfix = argv[1];
    options.row_index = static_cast<std::size_t>(std::stoull(argv[2]));

    for (int i = 3; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + flag);
            }
            return argv[++i];
        };

        if (key == "--population-size") {
            options.population_size = parse_int(need_value(key));
        } else if (key == "--num-payoff-permutations") {
            options.num_payoff_permutations = parse_int(need_value(key));
        } else if (key == "--max-social-steps") {
            options.max_social_steps = parse_int(need_value(key));
        } else if (key == "--target-omniscient-fraction") {
            options.target_omniscient_fraction = parse_double(need_value(key));
        } else if (key == "--initialization-reset-rate") {
            options.initialization_reset_rate = parse_double(need_value(key));
        } else if (key == "--reset-rate-search-min") {
            options.reset_rate_search_min = parse_double(need_value(key));
        } else if (key == "--reset-rate-search-max") {
            options.reset_rate_search_max = parse_double(need_value(key));
        } else if (key == "--reset-rate-search-iterations") {
            options.reset_rate_search_iterations = parse_int(need_value(key));
        } else if (key == "--reset-rate-tuning-population-size") {
            options.reset_rate_tuning_population_size = parse_int(need_value(key));
        } else if (key == "--m-demonstrators") {
            options.m_demonstrators = parse_int(need_value(key));
        } else if (key == "--beta-payoff") {
            options.beta_payoff = parse_double(need_value(key));
        } else if (key == "--beta-conformity") {
            options.beta_conformity = parse_double(need_value(key));
        } else if (key == "--fixed-reset-rate") {
            options.fixed_reset_rate = parse_double(need_value(key));
        } else if (key == "--seed") {
            options.rng_seed = parse_u64(need_value(key));
        } else if (key == "--num-threads") {
            options.num_threads = parse_int(need_value(key));
        } else if (key == "--convergence-window-multiplier") {
            options.convergence_window_multiplier = parse_int(need_value(key));
        } else if (key == "--convergence-tolerance") {
            options.convergence_tolerance = parse_double(need_value(key));
        } else if (key == "--convergence-patience") {
            options.convergence_patience = parse_int(need_value(key));
        } else if (key == "--min-ols-trait-rows") {
            options.min_ols_trait_rows = parse_int(need_value(key));
        } else if (key == "--min-ols-focal-repertoires") {
            options.min_ols_focal_repertoires = parse_int(need_value(key));
        } else if (key == "--min-payoff-estimates") {
            options.min_payoff_estimates = parse_int(need_value(key));
        } else if (key == "--social-learning-failure-free") {
            options.social_learning_failure_free = true;
        } else if (key == "--emit-detailed-results") {
            options.emit_detailed_results = true;
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }

    if (options.population_size <= 0) {
        throw std::runtime_error("--population-size must be positive");
    }
    if (options.num_payoff_permutations <= 0) {
        throw std::runtime_error("--num-payoff-permutations must be positive");
    }
    if (options.m_demonstrators <= 0) {
        throw std::runtime_error("--m-demonstrators must be positive");
    }
    if (options.max_social_steps < 0) {
        options.max_social_steps = 100 * options.population_size;
    }
    if (options.target_omniscient_fraction <= 0.0 || options.target_omniscient_fraction >= 1.0) {
        throw std::runtime_error("--target-omniscient-fraction must be between 0 and 1");
    }
    if (options.reset_rate_search_min <= 0.0 || options.reset_rate_search_min >= 1.0
        || options.reset_rate_search_max <= 0.0 || options.reset_rate_search_max >= 1.0
        || options.reset_rate_search_min >= options.reset_rate_search_max) {
        throw std::runtime_error("--reset-rate-search-min/max must satisfy 0 < min < max < 1");
    }
    if (options.reset_rate_search_iterations <= 0) {
        throw std::runtime_error("--reset-rate-search-iterations must be positive");
    }
    if (options.reset_rate_tuning_population_size <= 0) {
        throw std::runtime_error("--reset-rate-tuning-population-size must be positive");
    }
    if (options.initialization_reset_rate.has_value()
        && (*options.initialization_reset_rate <= 0.0 || *options.initialization_reset_rate >= 1.0)) {
        throw std::runtime_error("--initialization-reset-rate must be between 0 and 1");
    }
    if (options.fixed_reset_rate.has_value()
        && (*options.fixed_reset_rate < 0.0 || *options.fixed_reset_rate > 1.0)) {
        throw std::runtime_error("--fixed-reset-rate must be between 0 and 1");
    }
    if (options.min_ols_trait_rows < 3) {
        throw std::runtime_error("--min-ols-trait-rows must be at least 3");
    }
    if (options.min_ols_focal_repertoires < 1) {
        throw std::runtime_error("--min-ols-focal-repertoires must be at least 1");
    }
    if (options.min_payoff_estimates < 1) {
        throw std::runtime_error("--min-payoff-estimates must be at least 1");
    }

    return options;
}

[[nodiscard]] std::string selected_flattened_row(const std::string& postfix, std::size_t row_index) {
    std::string path = "adj_mat_" + postfix + ".csv";
    std::ifstream input(path);
    if (!input) {
        path = "input/adj_mat_" + postfix + ".csv";
        input.open(path);
    }
    if (!input) {
        throw std::runtime_error("Could not open input file: adj_mat_" + postfix + ".csv");
    }

    std::string line;
    for (std::size_t row = 0; std::getline(input, line); ++row) {
        if (row == row_index) {
            std::string bits;
            bits.reserve(line.size());
            for (const char ch : line) {
                if (ch == '0' || ch == '1') {
                    bits.push_back(ch);
                }
            }
            if (bits.empty()) {
                throw std::runtime_error("Selected row contains no 0/1 adjacency data");
            }
            return bits;
        }
    }

    throw std::runtime_error("Row index is outside input file: " + std::to_string(row_index));
}

[[nodiscard]] Graph parse_graph(const std::string& flattened) {
    const auto len = static_cast<int>(flattened.size());
    const auto n = static_cast<int>(std::llround(std::sqrt(len)));
    if (n * n != len) {
        throw std::runtime_error("Flattened adjacency length is not a perfect square");
    }
    if (n <= 0 || n > 64) {
        throw std::runtime_error("This implementation requires 1 <= N <= 64");
    }

    Graph graph;
    graph.n = n;
    graph.parents.assign(n, 0);
    graph.full_mask = n == 64 ? ~Mask{0} : ((Mask{1} << n) - 1);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const char ch = flattened[static_cast<std::size_t>(i * n + j)];
            if (ch == '1') {
                graph.parents[j] |= Mask{1} << i;
            } else if (ch != '0') {
                throw std::runtime_error("Adjacency data must contain only 0 and 1");
            }
        }
    }

    return graph;
}

[[nodiscard]] int popcount(Mask mask) {
    return std::popcount(mask);
}

[[nodiscard]] int nth_set_bit(Mask mask, int index) {
    for (int bit = 0; bit < 64; ++bit) {
        if ((mask & (Mask{1} << bit)) != 0) {
            if (index == 0) {
                return bit;
            }
            --index;
        }
    }
    throw std::runtime_error("nth_set_bit index outside mask");
}

void random_asocial_attempt(Mask& repertoire, const Graph& graph, std::mt19937_64& rng) {
    const Mask unlearned = graph.full_mask & ~repertoire;
    const int choices = popcount(unlearned);
    if (choices == 0) {
        return;
    }

    std::uniform_int_distribution<int> trait_choice(0, choices - 1);
    const int trait = nth_set_bit(unlearned, trait_choice(rng));
    if (graph.learnable(repertoire, trait)) {
        repertoire |= Mask{1} << trait;
    }
}

[[nodiscard]] Mask random_asocial_repertoire_after_age(
    const Graph& graph,
    int age,
    std::mt19937_64& rng
) {
    Mask repertoire = 0;
    for (int attempt = 0; attempt < age && repertoire != graph.full_mask; ++attempt) {
        random_asocial_attempt(repertoire, graph, rng);
    }
    return repertoire;
}

[[nodiscard]] PopulationSummary summarize_population(const std::vector<Mask>& agents, const Graph& graph) {
    PopulationSummary summary;
    int omniscient = 0;
    std::int64_t total_size = 0;
    for (const Mask repertoire : agents) {
        if (repertoire == graph.full_mask) {
            ++omniscient;
        }
        total_size += popcount(repertoire);
    }
    summary.omniscient_fraction = static_cast<double>(omniscient) / static_cast<double>(agents.size());
    summary.mean_repertoire_size = static_cast<double>(total_size) / static_cast<double>(agents.size());
    return summary;
}

[[nodiscard]] double evaluate_initialization_reset_rate(
    const Graph& graph,
    const Options& options,
    double reset_rate,
    std::uint64_t seed
) {
    int omniscient = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : omniscient) schedule(static)
#endif
    for (int i = 0; i < options.reset_rate_tuning_population_size; ++i) {
        std::mt19937_64 rng(splitmix64(seed + static_cast<std::uint64_t>(i)));
        std::geometric_distribution<int> age_dist(reset_rate);
        const Mask repertoire = random_asocial_repertoire_after_age(graph, age_dist(rng), rng);
        if (repertoire == graph.full_mask) {
            ++omniscient;
        }
    }

    return static_cast<double>(omniscient) / static_cast<double>(options.reset_rate_tuning_population_size);
}

[[nodiscard]] double tune_initialization_reset_rate(const Graph& graph, const Options& options) {
    double low = options.reset_rate_search_min;
    double high = options.reset_rate_search_max;
    for (int iter = 0; iter < options.reset_rate_search_iterations; ++iter) {
        const double mid = 0.5 * (low + high);
        const double fraction = evaluate_initialization_reset_rate(
            graph,
            options,
            mid,
            options.rng_seed + 10'000 + static_cast<std::uint64_t>(iter)
        );
        if (fraction > options.target_omniscient_fraction) {
            low = mid;
        } else {
            high = mid;
        }
    }
    return 0.5 * (low + high);
}

[[nodiscard]] std::vector<Mask> initialize_population(
    const Graph& graph,
    const Options& options,
    double reset_rate
) {
    std::vector<Mask> agents(static_cast<std::size_t>(options.population_size), 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < options.population_size; ++i) {
        std::mt19937_64 rng(splitmix64(options.rng_seed + 1 + static_cast<std::uint64_t>(i)));
        std::geometric_distribution<int> age_dist(reset_rate);
        agents[static_cast<std::size_t>(i)] = random_asocial_repertoire_after_age(graph, age_dist(rng), rng);
    }
    return agents;
}

[[nodiscard]] std::vector<double> baseline_payoffs(int n) {
    std::vector<double> payoffs(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        payoffs[static_cast<std::size_t>(i)] = 2.0 * static_cast<double>(i + 1) / static_cast<double>(n + 1);
    }
    return payoffs;
}

[[nodiscard]] std::uint64_t capped_factorial(int n, std::uint64_t cap) {
    std::uint64_t value = 1;
    for (int i = 2; i <= n; ++i) {
        if (value > cap / static_cast<std::uint64_t>(i)) {
            return cap;
        }
        value *= static_cast<std::uint64_t>(i);
    }
    return value;
}

[[nodiscard]] std::vector<std::vector<double>> payoff_assignments(const Graph& graph, const Options& options) {
    std::vector<std::vector<double>> assignments;
    assignments.reserve(static_cast<std::size_t>(options.num_payoff_permutations));
    assignments.push_back(baseline_payoffs(graph.n));

    const std::uint64_t max_unique = capped_factorial(graph.n, static_cast<std::uint64_t>(options.num_payoff_permutations));
    const int target = static_cast<int>(std::min<std::uint64_t>(
        static_cast<std::uint64_t>(options.num_payoff_permutations),
        max_unique
    ));

    std::mt19937_64 rng(options.rng_seed + 2);
    std::set<std::vector<double>> seen;
    seen.insert(assignments.front());
    auto next = assignments.front();

    while (static_cast<int>(assignments.size()) < target) {
        std::shuffle(next.begin(), next.end(), rng);
        if (seen.insert(next).second) {
            assignments.push_back(next);
        }
    }

    return assignments;
}

[[nodiscard]] std::unordered_map<Mask, int> repertoire_counts(const std::vector<Mask>& agents) {
    std::unordered_map<Mask, int> counts;
    counts.reserve(agents.size());
    for (const Mask repertoire : agents) {
        ++counts[repertoire];
    }
    return counts;
}

[[nodiscard]] double total_variation(
    const std::unordered_map<Mask, int>& current,
    const std::unordered_map<Mask, int>& previous,
    int population_size
) {
    double sum = 0.0;
    for (const auto& [rep, count] : current) {
        const auto found = previous.find(rep);
        const int old_count = found == previous.end() ? 0 : found->second;
        sum += std::abs(count - old_count);
    }
    for (const auto& [rep, count] : previous) {
        if (current.find(rep) == current.end()) {
            sum += count;
        }
    }
    return 0.5 * sum / static_cast<double>(population_size);
}

void social_learning_step(
    std::vector<Mask>& agents,
    const Graph& graph,
    const std::vector<double>& payoffs,
    Strategy strategy,
    double reset_rate,
    const Options& options,
    std::mt19937_64& rng
) {
    std::uniform_int_distribution<int> agent_dist(0, static_cast<int>(agents.size()) - 1);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    const int focal_index = agent_dist(rng);
    Mask& focal = agents[static_cast<std::size_t>(focal_index)];
    if (unit(rng) < reset_rate) {
        focal = 0;
        return;
    }

    std::vector<int> counts(static_cast<std::size_t>(graph.n), 0);
    for (int draw = 0; draw < options.m_demonstrators; ++draw) {
        const Mask demonstrator = agents[static_cast<std::size_t>(agent_dist(rng))];
        const int repertoire_size = popcount(demonstrator);
        if (repertoire_size == 0) {
            continue;
        }
        std::uniform_int_distribution<int> trait_choice(0, repertoire_size - 1);
        const int expressed = nth_set_bit(demonstrator, trait_choice(rng));
        if ((focal & (Mask{1} << expressed)) == 0) {
            ++counts[static_cast<std::size_t>(expressed)];
        }
    }

    std::vector<double> weights(static_cast<std::size_t>(graph.n), 0.0);
    double total_weight = 0.0;
    for (int trait = 0; trait < graph.n; ++trait) {
        const int count = counts[static_cast<std::size_t>(trait)];
        if (count == 0) {
            continue;
        }
        double weight = 0.0;
        if (strategy == Strategy::PayoffBiased) {
            weight = static_cast<double>(count)
                * std::pow(payoffs[static_cast<std::size_t>(trait)], options.beta_payoff);
        } else {
            weight = std::pow(static_cast<double>(count), options.beta_conformity);
        }
        weights[static_cast<std::size_t>(trait)] = weight;
        total_weight += weight;
    }

    if (total_weight <= 0.0) {
        return;
    }

    while (total_weight > 0.0) {
        std::uniform_real_distribution<double> weight_dist(0.0, total_weight);
        double cursor = weight_dist(rng);
        int selected = -1;
        for (int trait = 0; trait < graph.n; ++trait) {
            if (weights[static_cast<std::size_t>(trait)] <= 0.0) {
                continue;
            }
            cursor -= weights[static_cast<std::size_t>(trait)];
            if (cursor <= 0.0) {
                selected = trait;
                break;
            }
        }
        if (selected < 0) {
            return;
        }

        if (graph.learnable(focal, selected)) {
            focal |= Mask{1} << selected;
            return;
        }
        if (!options.social_learning_failure_free) {
            return;
        }

        const double selected_weight = weights[static_cast<std::size_t>(selected)];
        if (selected_weight <= 0.0) {
            return;
        }
        total_weight -= selected_weight;
        weights[static_cast<std::size_t>(selected)] = 0.0;
    }
}

[[nodiscard]] SimulationDiagnostics run_social_learning(
    std::vector<Mask>& agents,
    const Graph& graph,
    const std::vector<double>& payoffs,
    Strategy strategy,
    double reset_rate,
    const Options& options,
    std::uint64_t seed
) {
    std::mt19937_64 rng(seed);
    const int window = std::max(1, options.convergence_window_multiplier * options.population_size);
    auto previous = repertoire_counts(agents);
    int stable_windows = 0;
    double final_tv = std::numeric_limits<double>::quiet_NaN();
    int step = 0;

    for (; step < options.max_social_steps; ++step) {
        social_learning_step(agents, graph, payoffs, strategy, reset_rate, options, rng);
        if ((step + 1) % window == 0) {
            const auto current = repertoire_counts(agents);
            final_tv = total_variation(current, previous, options.population_size);
            stable_windows = final_tv < options.convergence_tolerance ? stable_windows + 1 : 0;
            previous = current;
            if (stable_windows >= options.convergence_patience) {
                ++step;
                break;
            }
        }
    }

    return {.steps_run = step, .final_tv_distance = final_tv};
}

[[nodiscard]] std::vector<double> trait_visibility(const std::vector<Mask>& agents, const Graph& graph) {
    std::vector<double> visibility(static_cast<std::size_t>(graph.n), 0.0);
    for (const Mask repertoire : agents) {
        const int repertoire_size = popcount(repertoire);
        if (repertoire_size == 0) {
            continue;
        }
        const double contribution = 1.0 / static_cast<double>(repertoire_size);
        for (int trait = 0; trait < graph.n; ++trait) {
            if ((repertoire & (Mask{1} << trait)) != 0) {
                visibility[static_cast<std::size_t>(trait)] += contribution;
            }
        }
    }
    for (double& value : visibility) {
        value /= static_cast<double>(agents.size());
    }
    return visibility;
}

struct OlsAccumulator {
    double n = 0.0;
    double sx1 = 0.0;
    double sx2 = 0.0;
    double sy = 0.0;
    double sx1x1 = 0.0;
    double sx1x2 = 0.0;
    double sx2x2 = 0.0;
    double sx1y = 0.0;
    double sx2y = 0.0;

    void add(double x1, double x2, double y) {
        n += 1.0;
        sx1 += x1;
        sx2 += x2;
        sy += y;
        sx1x1 += x1 * x1;
        sx1x2 += x1 * x2;
        sx2x2 += x2 * x2;
        sx1y += x1 * y;
        sx2y += x2 * y;
    }
};

[[nodiscard]] std::optional<std::pair<double, double>> solve_eta_chi(const OlsAccumulator& a) {
    if (a.n < 3.0) {
        return std::nullopt;
    }

    double mat[3][4] = {
        {a.n, a.sx1, a.sx2, a.sy},
        {a.sx1, a.sx1x1, a.sx1x2, a.sx1y},
        {a.sx2, a.sx1x2, a.sx2x2, a.sx2y},
    };

    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 3; ++row) {
            if (std::abs(mat[row][col]) > std::abs(mat[pivot][col])) {
                pivot = row;
            }
        }
        if (std::abs(mat[pivot][col]) < 1e-12) {
            return std::nullopt;
        }
        if (pivot != col) {
            for (int k = col; k < 4; ++k) {
                std::swap(mat[col][k], mat[pivot][k]);
            }
        }
        const double divisor = mat[col][col];
        for (int k = col; k < 4; ++k) {
            mat[col][k] /= divisor;
        }
        for (int row = 0; row < 3; ++row) {
            if (row == col) {
                continue;
            }
            const double factor = mat[row][col];
            for (int k = col; k < 4; ++k) {
                mat[row][k] -= factor * mat[col][k];
            }
        }
    }

    const double inaccessible_coef = mat[1][3];
    const double payoff_coef = mat[2][3];
    return std::make_pair(-inaccessible_coef, payoff_coef);
}

struct MetricAccumulator {
    OlsAccumulator ols;
    std::set<Mask> repertoires;
    int trait_rows = 0;
};

[[nodiscard]] std::vector<MetricRow> compute_metrics(
    const std::vector<Mask>& agents,
    const Graph& graph,
    const std::vector<double>& payoffs,
    const std::string& strategy_name,
    int payoff_assignment_id,
    const SimulationDiagnostics& diagnostics,
    const Options& options
) {
    const auto counts = repertoire_counts(agents);
    const auto visibility = trait_visibility(agents, graph);
    const auto summary = summarize_population(agents, graph);
    std::map<std::pair<int, int>, MetricAccumulator> by_bin;

    for (const auto& [repertoire, count] : counts) {
        (void)count;
        const Mask unlearned = graph.full_mask & ~repertoire;
        const int unlearned_count = popcount(unlearned);
        if (unlearned_count < 2) {
            continue;
        }

        Mask accessible = 0;
        for (int trait = 0; trait < graph.n; ++trait) {
            if ((unlearned & (Mask{1} << trait)) != 0 && graph.learnable(repertoire, trait)) {
                accessible |= Mask{1} << trait;
            }
        }
        const Mask inaccessible = unlearned & ~accessible;
        const int inaccessible_count = popcount(inaccessible);
        const int kappa_key = static_cast<int>(std::llround(
            1'000'000.0 * static_cast<double>(inaccessible_count) / static_cast<double>(unlearned_count)
        ));
        const int repertoire_size = popcount(repertoire);

        double visibility_sum = 0.0;
        double log_payoff_sum = 0.0;
        for (int trait = 0; trait < graph.n; ++trait) {
            if ((unlearned & (Mask{1} << trait)) != 0) {
                visibility_sum += visibility[static_cast<std::size_t>(trait)];
                log_payoff_sum += std::log(payoffs[static_cast<std::size_t>(trait)]);
            }
        }
        if (visibility_sum <= 0.0) {
            continue;
        }
        const double mean_log_payoff = log_payoff_sum / static_cast<double>(unlearned_count);
        auto& accumulator = by_bin[{kappa_key, repertoire_size}];
        accumulator.repertoires.insert(repertoire);

        for (int trait = 0; trait < graph.n; ++trait) {
            if ((unlearned & (Mask{1} << trait)) == 0) {
                continue;
            }
            const double normalized_visibility = visibility[static_cast<std::size_t>(trait)] / visibility_sum;
            if (normalized_visibility <= 0.0) {
                continue;
            }
            const double x_inaccessible = (inaccessible & (Mask{1} << trait)) != 0 ? 1.0 : 0.0;
            const double x_payoff = std::log(payoffs[static_cast<std::size_t>(trait)]) - mean_log_payoff;
            const double y = std::log(normalized_visibility);
            accumulator.ols.add(x_inaccessible, x_payoff, y);
            ++accumulator.trait_rows;
        }
    }

    std::vector<MetricRow> rows;
    for (const auto& [key, accumulator] : by_bin) {
        if (accumulator.trait_rows < options.min_ols_trait_rows
            || static_cast<int>(accumulator.repertoires.size()) < options.min_ols_focal_repertoires) {
            continue;
        }
        const auto estimate = solve_eta_chi(accumulator.ols);
        if (!estimate.has_value()) {
            continue;
        }
        const auto [eta, chi] = *estimate;
        rows.push_back({
            .strategy = strategy_name,
            .payoff_assignment_id = payoff_assignment_id,
            .local_kappa = static_cast<double>(key.first) / 1'000'000.0,
            .repertoire_size_bin = key.second,
            .eta = eta,
            .chi = chi,
            .num_focal_repertoires = static_cast<int>(accumulator.repertoires.size()),
            .num_trait_focal_rows = accumulator.trait_rows,
            .final_omniscient_fraction = summary.omniscient_fraction,
            .mean_repertoire_size = summary.mean_repertoire_size,
            .steps_run = diagnostics.steps_run,
            .final_tv_distance = diagnostics.final_tv_distance,
        });
    }
    return rows;
}

void append_rows(
    std::vector<MetricRow>& target,
    std::vector<MetricRow>&& source
) {
    target.insert(target.end(), std::make_move_iterator(source.begin()), std::make_move_iterator(source.end()));
}

void write_results(
    const std::string& path,
    const Options& options,
    double reset_rate,
    const std::vector<MetricRow>& rows
) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Could not write output file: " + path);
    }

    output
        << "postfix,row_index,payoff_assignment_id,resident_strategy,local_kappa,"
        << "repertoire_size_bin,eta,chi,num_focal_repertoires,num_trait_focal_rows,"
        << "reset_rate,beta_payoff,beta_conformity,m_demonstrators,population_size,"
        << "num_social_steps,rng_seed,social_learning_failure_free,final_omniscient_fraction,mean_repertoire_size,"
        << "final_tv_distance\n";

    output << std::setprecision(17);
    for (const auto& row : rows) {
        output
            << options.postfix << ','
            << options.row_index << ','
            << row.payoff_assignment_id << ','
            << row.strategy << ','
            << row.local_kappa << ','
            << row.repertoire_size_bin << ','
            << row.eta << ','
            << row.chi << ','
            << row.num_focal_repertoires << ','
            << row.num_trait_focal_rows << ','
            << reset_rate << ','
            << options.beta_payoff << ','
            << options.beta_conformity << ','
            << options.m_demonstrators << ','
            << options.population_size << ','
            << row.steps_run << ','
            << options.rng_seed << ','
            << options.social_learning_failure_free << ','
            << row.final_omniscient_fraction << ','
            << row.mean_repertoire_size << ','
            << row.final_tv_distance << '\n';
    }
}

[[nodiscard]] std::vector<AveragedMetricRow> average_over_payoffs(
    const std::vector<MetricRow>& rows,
    const Options& options
) {
    struct Accumulator {
        double sum_eta = 0.0;
        double sum_chi = 0.0;
        double sum_eta_sq = 0.0;
        double sum_chi_sq = 0.0;
        double sum_focal_repertoires = 0.0;
        double sum_trait_rows = 0.0;
        int count = 0;
    };

    std::map<std::tuple<std::string, int, int>, Accumulator> accumulators;
    for (const auto& row : rows) {
        const int kappa_key = static_cast<int>(std::llround(row.local_kappa * 1'000'000.0));
        auto& acc = accumulators[{row.strategy, kappa_key, row.repertoire_size_bin}];
        acc.sum_eta += row.eta;
        acc.sum_chi += row.chi;
        acc.sum_eta_sq += row.eta * row.eta;
        acc.sum_chi_sq += row.chi * row.chi;
        acc.sum_focal_repertoires += row.num_focal_repertoires;
        acc.sum_trait_rows += row.num_trait_focal_rows;
        ++acc.count;
    }

    std::vector<AveragedMetricRow> averaged;
    averaged.reserve(accumulators.size());
    for (const auto& [key, acc] : accumulators) {
        if (acc.count < options.min_payoff_estimates) {
            continue;
        }
        const auto& [strategy, kappa_key, repertoire_size] = key;
        const double count = static_cast<double>(acc.count);
        const double mean_eta = acc.sum_eta / count;
        const double mean_chi = acc.sum_chi / count;
        const double eta_var = acc.count > 1
            ? (acc.sum_eta_sq - count * mean_eta * mean_eta) / static_cast<double>(acc.count - 1)
            : 0.0;
        const double chi_var = acc.count > 1
            ? (acc.sum_chi_sq - count * mean_chi * mean_chi) / static_cast<double>(acc.count - 1)
            : 0.0;
        averaged.push_back({
            .strategy = strategy,
            .local_kappa = static_cast<double>(kappa_key) / 1'000'000.0,
            .repertoire_size_bin = repertoire_size,
            .mean_eta = mean_eta,
            .mean_chi = mean_chi,
            .sd_eta = std::sqrt(std::max(0.0, eta_var)),
            .sd_chi = std::sqrt(std::max(0.0, chi_var)),
            .num_payoff_estimates = acc.count,
            .mean_focal_repertoires = acc.sum_focal_repertoires / count,
            .mean_trait_focal_rows = acc.sum_trait_rows / count,
        });
    }
    return averaged;
}

void write_averaged_results(
    const std::string& path,
    const Options& options,
    double reset_rate,
    const std::vector<AveragedMetricRow>& rows
) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Could not write output file: " + path);
    }

    output
        << "postfix,row_index,resident_strategy,local_kappa,repertoire_size_bin,"
        << "mean_eta,mean_chi,sd_eta,sd_chi,num_payoff_estimates,"
        << "mean_focal_repertoires,mean_trait_focal_rows,reset_rate,beta_payoff,"
        << "beta_conformity,m_demonstrators,population_size,rng_seed,social_learning_failure_free\n";

    output << std::setprecision(17);
    for (const auto& row : rows) {
        output
            << options.postfix << ','
            << options.row_index << ','
            << row.strategy << ','
            << row.local_kappa << ','
            << row.repertoire_size_bin << ','
            << row.mean_eta << ','
            << row.mean_chi << ','
            << row.sd_eta << ','
            << row.sd_chi << ','
            << row.num_payoff_estimates << ','
            << row.mean_focal_repertoires << ','
            << row.mean_trait_focal_rows << ','
            << reset_rate << ','
            << options.beta_payoff << ','
            << options.beta_conformity << ','
            << options.m_demonstrators << ','
            << options.population_size << ','
            << options.rng_seed << ','
            << options.social_learning_failure_free << '\n';
    }
}

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
#ifdef _OPENMP
        if (options.num_threads > 0) {
            omp_set_num_threads(options.num_threads);
        }
#else
        if (options.num_threads > 0) {
            std::cerr << "Warning: OpenMP is not enabled; --num-threads ignored\n";
        }
#endif

        const Graph graph = parse_graph(selected_flattened_row(options.postfix, options.row_index));
        const double initialization_reset_rate = options.initialization_reset_rate.value_or(
            tune_initialization_reset_rate(graph, options)
        );
        const double social_reset_rate = options.fixed_reset_rate.value_or(initialization_reset_rate);
        const auto initialized = initialize_population(graph, options, initialization_reset_rate);
        const auto payoffs = payoff_assignments(graph, options);

        auto conformist_population = initialized;
        const auto conformist_diag = run_social_learning(
            conformist_population,
            graph,
            payoffs.front(),
            Strategy::Conformist,
            social_reset_rate,
            options,
            options.rng_seed + 200'000
        );

        std::vector<MetricRow> all_rows;

#ifdef _OPENMP
#pragma omp parallel
        {
            std::vector<MetricRow> local_rows;
#pragma omp for schedule(dynamic)
            for (int p = 0; p < static_cast<int>(payoffs.size()); ++p) {
                auto payoff_population = initialized;

                const auto payoff_diag = run_social_learning(
                    payoff_population,
                    graph,
                    payoffs[static_cast<std::size_t>(p)],
                    Strategy::PayoffBiased,
                    social_reset_rate,
                    options,
                    options.rng_seed + 100'000 + static_cast<std::uint64_t>(p)
                );

                append_rows(local_rows, compute_metrics(
                    payoff_population,
                    graph,
                    payoffs[static_cast<std::size_t>(p)],
                    "payoff_biased",
                    p,
                    payoff_diag,
                    options
                ));
                append_rows(local_rows, compute_metrics(
                    conformist_population,
                    graph,
                    payoffs[static_cast<std::size_t>(p)],
                    "conformist",
                    p,
                    conformist_diag,
                    options
                ));
            }
#pragma omp critical
            append_rows(all_rows, std::move(local_rows));
        }
#else
        for (int p = 0; p < static_cast<int>(payoffs.size()); ++p) {
            auto payoff_population = initialized;

            const auto payoff_diag = run_social_learning(
                payoff_population,
                graph,
                payoffs[static_cast<std::size_t>(p)],
                Strategy::PayoffBiased,
                social_reset_rate,
                options,
                options.rng_seed + 100'000 + static_cast<std::uint64_t>(p)
            );

            append_rows(all_rows, compute_metrics(
                payoff_population,
                graph,
                payoffs[static_cast<std::size_t>(p)],
                "payoff_biased",
                p,
                payoff_diag,
                options
            ));
            append_rows(all_rows, compute_metrics(
                conformist_population,
                graph,
                payoffs[static_cast<std::size_t>(p)],
                "conformist",
                p,
                conformist_diag,
                options
            ));
        }
#endif

        std::sort(all_rows.begin(), all_rows.end(), [](const MetricRow& left, const MetricRow& right) {
            return std::tie(left.payoff_assignment_id, left.strategy, left.repertoire_size_bin, left.local_kappa)
                < std::tie(right.payoff_assignment_id, right.strategy, right.repertoire_size_bin, right.local_kappa);
        });

        std::filesystem::create_directories("output");
        const std::string output_path = "output/results_" + options.postfix + "_"
            + std::to_string(options.row_index) + ".csv";
        const std::string averaged_output_path = "output/results_avg_" + options.postfix + "_"
            + std::to_string(options.row_index) + ".csv";
        if (options.emit_detailed_results) {
            write_results(output_path, options, social_reset_rate, all_rows);
        }
        write_averaged_results(averaged_output_path, options, social_reset_rate, average_over_payoffs(all_rows, options));

        const auto initialized_summary = summarize_population(initialized, graph);
        std::cerr
            << "N=" << graph.n
            << " initialization_reset_rate=" << initialization_reset_rate
            << " social_reset_rate=" << social_reset_rate
            << " initialized_omniscient_fraction=" << initialized_summary.omniscient_fraction
            << " payoff_assignments=" << payoffs.size()
            << " social_learning_failure_free=" << options.social_learning_failure_free
            << " rows=" << all_rows.size()
            << " detailed_output=" << (options.emit_detailed_results ? output_path : "not_emitted")
            << " averaged_output=" << averaged_output_path << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
