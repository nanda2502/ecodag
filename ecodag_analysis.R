if (rstudioapi::isAvailable()) setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


library(tidyverse)
library(patchwork)

df <- read_csv("results_avg_8_combined.csv")

df <- df %>%
    mutate(
        resident_strategy = recode(resident_strategy,
                                   conformist = "Conformity",
                                   payoff_biased = "Payoff bias"
        )
    ) %>% 
    filter(mean_trait_focal_rows >= 10)

# To do: switch to using the raw results and do uncertainty quantification correctly

plot_df <- df %>%
    group_by(resident_strategy, local_kappa) %>%
    summarise(
        eta = mean(mean_eta, na.rm = TRUE),
        chi = mean(mean_chi, na.rm = TRUE),
        se_eta = sqrt(sum(sd_eta^2, na.rm = TRUE)) / n(),
        se_chi = sqrt(sum(sd_chi^2, na.rm = TRUE)) / n(),
        .groups = "drop"
    ) %>%
    mutate(
        resident_strategy = factor(
            resident_strategy,
            levels = c("Conformity", "Payoff bias")
        )
    )

strategy_colors <- c(
    "Conformity"   = "#80B1E0",
    "Payoff bias"  = "#006328"
)

p_eta <- ggplot(plot_df, aes(local_kappa, eta, color = resident_strategy, fill = resident_strategy)) +
    geom_ribbon(aes(ymin = eta - se_eta, ymax = eta + se_eta), alpha = 0.2, color = NA) +
    geom_line() +
    geom_point() +
    scale_color_manual(values = strategy_colors) +
    scale_fill_manual(values = strategy_colors) +
    labs(
        x = NULL,
        y = expression("Accessibility filtering (" * eta * ")"),
        color = NULL,
        fill  = NULL
    ) +
    coord_cartesian(ylim = c(0, 7)) +
    theme_classic()

p_chi <- ggplot(plot_df, aes(local_kappa, chi, color = resident_strategy, fill = resident_strategy)) +
    geom_ribbon(aes(ymin = chi - se_chi, ymax = chi + se_chi), alpha = 0.2, color = NA) +
    geom_line() +
    geom_point() +
    scale_color_manual(values = strategy_colors) +
    scale_fill_manual(values = strategy_colors) +
    labs(
        x = expression("Constraint strength (" * kappa * ")"),
        y = expression("Payoff visibility (" * chi * ")"),
        color = NULL,
        fill  = NULL
    ) +
    coord_cartesian(ylim = c(-0.05, 1)) +
    theme_classic()

p_eta / p_chi

print(range(df$local_kappa, na.rm = TRUE))
print(range(df$mean_eta, na.rm = TRUE))
print(range(df$mean_chi, na.rm = TRUE))
print(range(df$sd_eta, na.rm = TRUE))
print(range(df$sd_chi, na.rm = TRUE))

df %>%
    group_by(resident_strategy) %>%
    summarise(
        kappa_min = min(local_kappa, na.rm = TRUE),
        kappa_max = max(local_kappa, na.rm = TRUE),
        eta_min = min(mean_eta, na.rm = TRUE),
        eta_max = max(mean_eta, na.rm = TRUE),
        chi_min = min(mean_chi, na.rm = TRUE),
        chi_max = max(mean_chi, na.rm = TRUE),
        .groups = "drop"
    ) %>%
    print(n = Inf)

df %>%
    group_by(resident_strategy, local_kappa) %>%
    summarise(
        eta = mean(mean_eta, na.rm = TRUE),
        chi = mean(mean_chi, na.rm = TRUE),
        n = n(),
        .groups = "drop"
    ) %>%
    arrange(resident_strategy, local_kappa) %>%
    print(n = Inf)



eta_diff <- df %>%
    group_by(local_kappa, repertoire_size_bin, resident_strategy) %>%
    summarise(
        mean_eta = mean(mean_eta, na.rm = TRUE),
        mean_trait_focal_rows = mean(mean_trait_focal_rows, na.rm = TRUE),
        mean_focal_repertoires = mean(mean_focal_repertoires, na.rm = TRUE),
        .groups = "drop"
    ) %>%
    pivot_wider(
        names_from = resident_strategy,
        values_from = c(mean_eta, mean_trait_focal_rows, mean_focal_repertoires)
    )


eta_diff <- eta_diff %>%
    mutate(diff_eta = `mean_eta_Payoff bias` - mean_eta_Conformity)

eta_diff %>%
    summarise(
        mean_diff = mean(diff_eta, na.rm = TRUE),
        median_diff = median(diff_eta, na.rm = TRUE),
        prop_payoff_higher = mean(diff_eta > 0, na.rm = TRUE),
        n = sum(!is.na(diff_eta))
    )

eta_diff %>%
    group_by(repertoire_size_bin) %>%
    summarise(
        mean_diff = mean(diff_eta, na.rm = TRUE),
        median_diff = median(diff_eta, na.rm = TRUE),
        prop_payoff_higher = mean(diff_eta > 0, na.rm = TRUE),
        n = sum(!is.na(diff_eta)),
        .groups = "drop"
    ) %>%
    print(n = Inf)


