# Load necessary libraries
library(ggplot2)
library(dplyr)
library(viridis)
library(ggpubr)
library(FSA)

# Define folder paths for females and males
female_folder_path <- "/Users/zariageorge/Desktop/WNPLAY/6.05.playlist/Females"
male_folder_path <- "/Users/zariageorge/Desktop/WNPLAY/6.05.playlist/Males"

# Get all CSV files from both folders
female_csv_files <- list.files(female_folder_path, pattern = "\\.csv$", full.names = TRUE)
male_csv_files <- list.files(male_folder_path, pattern = "\\.csv$", full.names = TRUE)

# Define original time windows (stimulus onset times)
time_windows <- list(
  Aggression = list(c(61.302,90.813), c(691.400,720.914), c(871.504,901.015),
                    c(961.503,991.014), c(1231.604,1261.114), c(1411.604,1441.113)),
  Distress   = list(c(151.302,180.851), c(241.303,270.072), c(601.403,630.953),
                    c(1051.501,1081.052), c(1951.803,1981.352), c(2041.804,2071.353)),
  Echolocation = list(c(421.402,450.44), c(511.746,540.944), c(1321.602,1351.145),
                      c(1501.602,1531.146), c(1771.803,1801.345), c(1861.803,1891.345)),
  Distortion = list(c(2131.428,2161.605), c(331.302,361.104), c(781.401,811.205),
                    c(1141.503,1171.305), c(1591.603,1621.406), c(1681.703,1711.504))
)

alternative_time_windows <- list(
  Aggression   = list(c(60.189,90.189), c(150.189,180.189), c(510.389,540.389),
                      c(1590.790,1620.790), c(1770.790,1800.790), c(2130.791,2160.791)),
  Distress     = list(c(420.389,450.389), c(600.389,630.389), c(780.489,810.489),
                      c(870.489,900.489), c(1050.490,1080.490), c(1950.790,1980.790)),
  Echolocation = list(c(960.490,990.490), c(1320.690,1350.690), c(1500.790,1530.790),
                      c(1680.790,1710.790), c(1860.790,1890.790), c(2040.790,2070.790)),
  Distortion   = list(c(691.400,720.914), c(1141.503,1171.305), c(1231.604,1261.114),
                      c(1411.604,1441.113), c(241.303,270.072), c(331.302,361.104))
)

# Function to get the appropriate time windows for a bat
get_time_windows <- function(bat_id) {
  if (bat_id %in% c("AF4", "AF6", "B32", "AD6")) {
    return(alternative_time_windows)
  } else {
    return(time_windows)
  }
}

# Function to calculate robust baseline heart rate
calculate_robust_baseline <- function(df) {
  valid_rows <- df$Heart_Rate > 0 & df$Heart_Rate <= 350

  valid_hr <- df$Heart_Rate[valid_rows]
  
  if (length(valid_hr) < 5) {
    return(NA)
  }
  
  window_size <- 30
  min_avg_hr <- Inf
  
  time_range <- seq(min(df$Time), max(df$Time) - window_size, by = 5)
  
  for (start_time in time_range) {
    window_rows <- df$Time >= start_time & 
      df$Time <= start_time + window_size &
      df$Heart_Rate > 0 &
      df$Heart_Rate <= 310
    window_hr <- df$Heart_Rate[window_rows]
    
    if (length(window_hr) >= 5) {
      avg_hr <- mean(window_hr)
      if (avg_hr < min_avg_hr) {
        min_avg_hr <- avg_hr
      }
    }
  }
  
  percentile_baseline <- quantile(valid_hr, 0.05)
  
  if (min_avg_hr < 50 || is.infinite(min_avg_hr)) {
    return(percentile_baseline)
  }
  
  return(min(min_avg_hr, percentile_baseline))
}

# Initialize lists to store data
plot_data <- list()
baseline_comparison <- list()

# Process all files
for (file_info in list(list(files = female_csv_files, gender = "Female"),
                       list(files = male_csv_files, gender = "Male"))) {
  
  csv_files <- file_info$files
  gender_label <- file_info$gender
  
  for (file in csv_files) {
    tryCatch({
      df <- read.csv(file)
      file_name <- basename(file)
      bat_id <- substr(file_name, 1, 3)
      
      # Calculate original baseline
      baseline_rows <- df$File_Marker == 0 & df$Heart_Rate > 0 & df$Heart_Rate <= 310
      original_baseline <- mean(df$Heart_Rate[baseline_rows], na.rm = TRUE)
      
      # Calculate robust baseline
      robust_baseline <- calculate_robust_baseline(df)
      
      if (!is.na(robust_baseline)) {
        baseline_hr <- robust_baseline
        baseline_method <- "robust"
      } else {
        baseline_hr <- original_baseline
        baseline_method <- "original"
      }
      
      # Record baseline comparison
      baseline_comparison[[length(baseline_comparison) + 1]] <- data.frame(
        Bat = bat_id,
        File = file_name,
        Original_Baseline = original_baseline,
        Robust_Baseline = baseline_hr,
        Method_Used = baseline_method,
        Difference = ifelse(baseline_method != "original", original_baseline - baseline_hr, 0)
      )
      
      # Normalize time
      marker_rows <- df$File_Marker == 1
      if (any(marker_rows)) {
        first_marker_time <- min(df$Time[marker_rows], na.rm = TRUE)
        df$Time <- df$Time - first_marker_time
      }
      
      # Get appropriate time windows for this bat
      bat_time_windows <- get_time_windows(bat_id)
      
      # Process each category
      for (category in names(bat_time_windows)) {
        intervals <- bat_time_windows[[category]]
        hr_values <- c()
        
        for (interval in intervals) {
          start <- interval[1]
          end <- interval[2]
          
          segment_rows <- df$Time >= start & df$Time <= end & 
            df$Heart_Rate > 0 & df$Heart_Rate <= 500
          hr_segment <- df$Heart_Rate[segment_rows]
          
          hr_values <- c(hr_values, hr_segment)
        }
        
        if (length(hr_values) > 0) {
          avg_hr <- mean(hr_values)
          change_from_baseline <- avg_hr - baseline_hr
          
          plot_data[[length(plot_data) + 1]] <- data.frame(
            Bat_ID = bat_id,
            Stimulus = category,
            Change_from_Baseline = change_from_baseline,
            Sex = gender_label,
            Avg_Heart_Rate = avg_hr,
            Baseline_HR = baseline_hr
          )
        }
      }
      
    }, error = function(e) {
      message(paste("Error processing file", file, ":", e$message))
    })
  }
}

# Convert lists to data frames
HRData_Clean <- do.call(rbind, plot_data)
baseline_df <- do.call(rbind, baseline_comparison)

# Print summary of baseline differences
cat("\nBaseline Comparison Summary:\n")
cat(sprintf("Average difference between original and robust baseline: %.2f BPM\n", 
            mean(baseline_df$Difference, na.rm = TRUE)))
cat(sprintf("Max difference: %.2f BPM\n", 
            max(baseline_df$Difference, na.rm = TRUE)))

# Filter data
filter_rows <- HRData_Clean$Change_from_Baseline > 0 & 
  HRData_Clean$Change_from_Baseline <= 300
HRData_Clean <- HRData_Clean[filter_rows, ]

# Summary statistics using aggregate
HR_summary <- aggregate(
  Change_from_Baseline ~ Stimulus,
  data = HRData_Clean,
  FUN = function(x) {
    c(n = length(x),
      Mean_Change = mean(x, na.rm = TRUE),
      SD_Change = sd(x, na.rm = TRUE),
      Median_Change = median(x, na.rm = TRUE),
      Min_Change = min(x, na.rm = TRUE),
      Max_Change = max(x, na.rm = TRUE))
  }
)
print(HR_summary)

# Check distribution visually
qqnorm(HRData_Clean$Change_from_Baseline, main = "QQ Plot")
qqline(HRData_Clean$Change_from_Baseline, col = "red", lwd = 2)

hist(HRData_Clean$Change_from_Baseline,
     main = "Histogram of HR Change from Baseline",
     xlab = "Change in Heart Rate (BPM)",
     col = "blue", border = "white")

# Plotting - Boxplot of change from baseline by stimulus
ggplot(HRData_Clean, aes(x = Stimulus, y = Change_from_Baseline, fill = Stimulus)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.1, alpha = 0.4, size = 2) +
  labs(
    title = "Change in Heart Rate from Baseline by Stimulus",
    x = "Stimulus Category",
    y = "Δ Heart Rate (BPM)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold")
  ) +
  scale_fill_viridis_d(option = "rocket")

# Calculate mean change for each bat within each stimulus
Bat_Means_Overall <- aggregate(
  Change_from_Baseline ~ Bat_ID + Stimulus,
  data = HRData_Clean,
  FUN = function(x) mean(x, na.rm = TRUE)
)
colnames(Bat_Means_Overall)[3] <- "Mean_Change_per_Bat"

# Box plot with mean as points
ggplot(HRData_Clean, aes(x = Stimulus, y = Change_from_Baseline, fill = Stimulus)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_point(
    data = Bat_Means_Overall,
    aes(x = Stimulus, y = Mean_Change_per_Bat),
    size = 3,
    alpha = 0.8,
    color = "black"
  ) +
  labs(
    title = "Change in Heart Rate from Baseline by Stimulus",
    x = "Stimulus Category",
    y = "Δ Heart Rate (BPM)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold")
  ) +
  scale_fill_viridis_d(option = "rocket")

# Non-parametric test: Kruskal-Wallis
kruskal_test <- kruskal.test(Change_from_Baseline ~ Stimulus, data = HRData_Clean)
print(kruskal_test)

# Add Sex variable
HRData_Clean$Sex <- NA
HRData_Clean$Sex[HRData_Clean$Bat_ID %in% c("A88", "AEO", "B05", "B1D", "B07","AD6","AF4")] <- "Male"
HRData_Clean$Sex[HRData_Clean$Bat_ID %in% c("693", "B04", "CD3", "B30","B32",'AF6')] <- "Female"

# Remove rows with NA Sex
HRData_Clean <- HRData_Clean[!is.na(HRData_Clean$Sex), ]

# Mann-Whitney U test (Wilcoxon test)
wilcox_test_result <- wilcox.test(Change_from_Baseline ~ Sex, data = HRData_Clean)
print(wilcox_test_result)

ggplot(HRData_Clean, aes(x = Sex, y = Change_from_Baseline, fill = Sex)) +
  geom_boxplot(alpha = 0.4, outlier.shape = NA) +
  geom_jitter(
    aes(color = Sex),
    position = position_jitterdodge(jitter.width = 0.6, dodge.width = 0.8),
    size = 2,
    alpha = 0.4
  ) +
  stat_compare_means(
    method = "t.test",
    comparisons = list(c("Male", "Female")),
    label = "p.signif",
    bracket.size = 0.8,
    label.y = max(HRData_Clean$Change_from_Baseline, na.rm = TRUE) + 5
  ) +
  labs(
    title = "Overall Change in Heart Rate by Sex",
    x = "Sex",
    y = "Δ Heart Rate (BPM)"
  ) +
  theme_minimal(base_size = 14) +
  scale_fill_manual(values = c("Male" = "#FFA500", "Female" = "#800080")) +
  scale_color_manual(values = c("Male" = "#FFA500", "Female" = "#800080")) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# Calculate means by bat and sex
Bat_Means_Overall_Sex <- aggregate(
  Change_from_Baseline ~ Bat_ID + Sex,
  data = HRData_Clean,
  FUN = function(x) mean(x, na.rm = TRUE)
)
colnames(Bat_Means_Overall_Sex)[3] <- "Mean_Change_per_Bat"

#median 

Bat_Medians_Overall_Sex <- aggregate(
  Change_from_Baseline ~ Bat_ID + Sex,
  data = HRData_Clean,
  FUN = function(x) median(x, na.rm = TRUE)
)
colnames(Bat_Medians_Overall_Sex)[3] <- "Median_Change_per_Bat"

ggplot(HRData_Clean, aes(x = Sex, y = Change_from_Baseline, fill = Sex, width = 0.3)) +
  geom_boxplot(alpha = 0.4, outlier.shape = NA) +
  geom_point(
    data = Bat_Medians_Overall_Sex,        # fixed: was Bat_Means_Overall_Sex
    aes(x = Sex, y = Median_Change_per_Bat, color = Sex),  # fixed: was Bat_Medians_Overall_Sex
    size = 3,
    alpha = 0.8
  ) +
  stat_compare_means(
    method = "wilcox.test",
    comparisons = list(c("Male", "Female")),
    label = "p.signif",
    bracket.size = 0.8,
    label.y = max(HRData_Clean$Change_from_Baseline, na.rm = TRUE) + 5
  ) +
  labs(
    title = "Overall Change in Heart Rate by Sex",
    x = "Sex",
    y = "Δ Heart Rate (BPM)"
  ) +
  theme_minimal(base_size = 14) +
  scale_fill_manual(values = c("Male" = "#FFA500", "Female" = "#800080")) +
  scale_color_manual(values = c("Male" = "#FFA500", "Female" = "#800080")) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))


# Custom function to format p-values
p_value_formatter <- function(p) {
  # Determine significance stars
  stars <- case_when(
    p < 0.001 ~ "***",
    p < 0.01 ~ "**",
    p < 0.05 ~ "*",
    TRUE ~ ""
  )
  
  # Format the p-value
  if (p < 0.001) {
    p_text <- "< 0.001"
  } else {
    p_text <- formatC(p, format = "f", digits = 3)
  }
  
  # Combine p-value and stars
  if (stars != "") {
    return(paste0(p_text, " ", stars))
  } else {
    return(p_text)
  }
}

# Box plot comparing Change in Heart Rate by Sex and Stimulus
ggplot(HRData_Clean, aes(x = Stimulus, y = Change_from_Baseline, fill = Sex)) +
  geom_boxplot(
    width = 0.5,
    position = position_dodge(width = 0.8),
    outlier.shape = NA,
    alpha = 0.7
  ) +
  geom_jitter(
    aes(color = Sex),  # Use Sex for color mapping
    position = position_jitterdodge(jitter.width = 0.6, dodge.width = 0.8),
    size = 2,
    alpha = 0.4
  ) +
  stat_compare_means(
    aes(group = Sex),  # Group by Sex for comparisons
    method = "wilcox.test",  # Use Mann-Whitney U test
    label = "p.format",  # Use the formatted p-value
    label.y = max(HRData_Clean$Change_from_Baseline, na.rm = TRUE) + 5,
    size = 5,
    digits = 2,
    p.adjust.method = "none"  # Adjust this if necessary
  ) +
  labs(
    title = "Change in Heart Rate by Sex and Stimulus",
    x = "Stimulus",
    y = "Δ Heart Rate (BPM)",
    fill = "Sex"
  ) +
  theme_minimal(base_size = 14) +
  scale_fill_manual(values = c("Male" = "#FFA500", "Female" = "#800080")) +  # Box colors
  scale_color_manual(values = c("Male" = "#FFA500", "Female" = "#800080")) +  # Jitter colors
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))


Bat_Medians <- aggregate(
  Change_from_Baseline ~ Bat_ID + Sex + Stimulus,
  data = HRData_Clean,
  FUN = function(x) median(x, na.rm = TRUE)
)
colnames(Bat_Medians)[4] <- "Median_Change_per_Bat"

# Box plot with medians
ggplot(HRData_Clean, aes(x = Stimulus, y = Change_from_Baseline, fill = Sex)) +
  geom_boxplot(
    width = 0.5,
    position = position_dodge(width = 0.8),
    outlier.shape = NA,
    alpha = 0.7
  ) +
  geom_point(
    data = Bat_Medians,
    aes(x = Stimulus, y = Median_Change_per_Bat, color = Sex),
    position = position_dodge(width = 0.8),
    size = 3,
    alpha = 0.8
  ) +
  stat_compare_means(
    aes(group = Sex),
    method = "wilcox.test",
    label = "p.signif",
    label.y = max(HRData_Clean$Change_from_Baseline, na.rm = TRUE) + 2
  ) +
  labs(
    title = "Change in Heart Rate by Sex and Stimulus",
    x = "Stimulus",
    y = "Δ Heart Rate (BPM)",
    fill = "Sex",
    color = "Sex"
  ) +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
  scale_fill_manual(values = c("Male" = "#FF8C00", "Female" = "#8A4D76")) +
  scale_color_manual(values = c("Male" = "#FF8C00", "Female" = "#8A4D76"))

#individual sex 
# Separate data by sex
Male_Data <- HRData_Clean[HRData_Clean$Sex == "Male", ]
Female_Data <- HRData_Clean[HRData_Clean$Sex == "Female", ]

# Non-parametric test: Kruskal-Wallis for Males
male_kruskal <- kruskal.test(Change_from_Baseline ~ Stimulus, data = Male_Data)
cat("\n=== Kruskal-Wallis Test for Males ===\n")
print(male_kruskal)

# Non-parametric test: Kruskal-Wallis for Females
female_kruskal <- kruskal.test(Change_from_Baseline ~ Stimulus, data = Female_Data)
cat("\n=== Kruskal-Wallis Test for Females ===\n")
print(female_kruskal)

# Post-hoc tests if significant (Dunn's test for non-parametric)
if (male_kruskal$p.value < 0.05) {
  cat("\n=== Dunn's Post-hoc Test for Males ===\n")
  male_posthoc <- dunnTest(Change_from_Baseline ~ Stimulus, data = Male_Data, method = "bh")
  print(male_posthoc)
}

if (female_kruskal$p.value < 0.05) {
  cat("\n=== Dunn's Post-hoc Test for Females ===\n")
  female_posthoc <- dunnTest(Change_from_Baseline ~ Stimulus, data = Female_Data, method = "bh")
  print(female_posthoc)
}

# Calculate means for males and females
Male_Means <- aggregate(
  Change_from_Baseline ~ Bat_ID + Stimulus,
  data = Male_Data,
  FUN = function(x) mean(x, na.rm = TRUE)
)
colnames(Male_Means)[3] <- "Mean_Change_per_Bat"

Female_Means <- aggregate(
  Change_from_Baseline ~ Bat_ID + Stimulus,
  data = Female_Data,
  FUN = function(x) mean(x, na.rm = TRUE)
)
colnames(Female_Means)[3] <- "Mean_Change_per_Bat"

# Plot for Males
male_plot <- ggplot(Male_Data, aes(x = Stimulus, y = Change_from_Baseline, fill = Stimulus)) +
  geom_boxplot(
    alpha = 0.7,
    outlier.shape = NA
  ) +
  geom_point(
    data = Male_Means,
    aes(x = Stimulus, y = Mean_Change_per_Bat),
    size = 3,
    alpha = 0.8,
    color = "black"
  ) +
  stat_compare_means(
    method = "kruskal.test",  # Use Kruskal-Wallis for comparison
    label.y = max(Male_Data$Change_from_Baseline, na.rm = TRUE) + 10
  ) +
  labs(
    title = "Males: Change in Heart Rate by Stimulus",
    x = "Stimulus Category",
    y = "Δ Heart Rate (BPM)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold")
  ) +
  scale_fill_manual(values = rep("#FFA500", 4))

# Plot for Females
female_plot <- ggplot(Female_Data, aes(x = Stimulus, y = Change_from_Baseline, fill = Stimulus)) +
  geom_boxplot(
    alpha = 0.7,
    outlier.shape = NA
  ) +
  geom_point(
    data = Female_Means,
    aes(x = Stimulus, y = Mean_Change_per_Bat),
    size = 3,
    alpha = 0.8,
    color = "black"
  ) +
  stat_compare_means(
    method = "kruskal.test",  # Use Kruskal-Wallis for comparison
    label.y = max(Female_Data$Change_from_Baseline, na.rm = TRUE) + 10
  ) +
  labs(
    title = "Females: Change in Heart Rate by Stimulus",
    x = "Stimulus Category",
    y = "Δ Heart Rate (BPM)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold")
  ) +
  scale_fill_manual(values = rep("#800080", 4))

# Display plots
print(male_plot)
print(female_plot)

# Combine plots
combined_plot <- ggarrange(
  male_plot,
  female_plot,
  ncol = 2,
  nrow = 1,
  labels = c("A", "B")
)
print(combined_plot)

# Summary statistics by sex and stimulus
summary_by_sex_stim <- aggregate(
  Change_from_Baseline ~ Sex + Stimulus,
  data = HRData_Clean,
  FUN = function(x) {
    c(n = length(x),
      Mean_Change = mean(x, na.rm = TRUE),
      SD_Change = sd(x, na.rm = TRUE),
      SE_Change = sd(x, na.rm = TRUE) / sqrt(length(x)))
  }
)
print(summary_by_sex_stim)



library(ggplot2)
library(ggpubr)
library(dplyr)

# Create grouped category variable
HRData_Clean$Category <- NA
HRData_Clean$Category[HRData_Clean$Stimulus %in% c("Aggression", "Distress")] <- "Social"
HRData_Clean$Category[HRData_Clean$Stimulus %in% c("Echolocation", "Distortion")] <- "Non-Social"

# Remove rows without a category
Category_Data <- HRData_Clean[!is.na(HRData_Clean$Category), ]

# Per-bat means per category
Category_Bat_Means <- aggregate(
  Change_from_Baseline ~ Bat_ID + Category,
  data = Category_Data,
  FUN = function(x) mean(x, na.rm = TRUE)
)
colnames(Category_Bat_Means)[3] <- "Mean_Change_per_Bat"

# Y-axis limit for sig bar
ymax <- max(Category_Data$Change_from_Baseline, na.rm = TRUE)


# Linear Mixed Effect Trial 
library(ggplot2)
library(ggpubr)
library(dplyr)
library(lme4)
library(lmerTest) 
# ── Subset: exclude Distortion ────────────────────────────────────────────────
SocNonSoc_Data <- HRData_Clean[
  HRData_Clean$Stimulus %in% c("Aggression", "Distress", "Echolocation"), 
]

SocNonSoc_Data$Category <- ifelse(
  SocNonSoc_Data$Stimulus %in% c("Aggression", "Distress"), "Social", "Non-Social"
)

# ── CRITICAL FIX: collapse to one median per bat per category FIRST ───────────
# This averages Aggression + Distress into a single Social value per bat
# so each bat contributes exactly ONE observation per category
Bat_Medians <- aggregate(
  Change_from_Baseline ~ Bat_ID + Sex + Category,
  data = SocNonSoc_Data,
  FUN  = function(x) median(x, na.rm = TRUE)
)
colnames(Bat_Medians)[4] <- "Median_Change"

# Verify: each bat should appear exactly twice (once Social, once Non-Social)
cat("\n=== Observations per bat (should be 2 each) ===\n")
print(table(Bat_Medians$Bat_ID))

# Set reference levels
Bat_Medians$Category <- factor(Bat_Medians$Category, levels = c("Non-Social", "Social"))
Bat_Medians$Sex      <- factor(Bat_Medians$Sex,      levels = c("Female", "Male"))

# ── LME on per-bat medians (balanced: 1 obs per bat per category) ─────────────
lme_model <- lmer(
  Median_Change ~ Category * Sex + (1 | Bat_ID),
  data = Bat_Medians
)

cat("\n=== Linear Mixed Effects Model ===\n")
print(summary(lme_model))

cat("\n=== ANOVA table (Type III) — interaction test ===\n")
print(anova(lme_model))

# Extract interaction p-value
lme_anova     <- anova(lme_model)
interaction_p <- lme_anova["Category:Sex", "Pr(>F)"]
int_p_label   <- ifelse(interaction_p < 0.001, "p < 0.001",
                        sprintf("p = %.3f", interaction_p))
int_sig       <- ifelse(interaction_p < 0.001, "***",
                        ifelse(interaction_p < 0.01,  "**",
                               ifelse(interaction_p < 0.05,  "*", "ns")))

cat(sprintf("\nSex × Category interaction: %s (%s)\n", int_sig, int_p_label))

# ── Group medians and SE for plots ────────────────────────────────────────────
Group_Stats <- aggregate(
  Median_Change ~ Sex + Category,
  data = Bat_Medians,
  FUN  = function(x) c(
    Median = median(x),
    SE     = sd(x) / sqrt(length(x))
  )
)
Group_Stats <- do.call(data.frame, Group_Stats)
colnames(Group_Stats) <- c("Sex", "Category", "Median", "SE")
Group_Stats$Category  <- factor(Group_Stats$Category, levels = c("Non-Social", "Social"))

# ── Sex difference per category for bar plot ──────────────────────────────────
male_med   <- Group_Stats[Group_Stats$Sex == "Male",   c("Category", "Median")]
female_med <- Group_Stats[Group_Stats$Sex == "Female", c("Category", "Median")]
colnames(male_med)[2]   <- "Male_Median"
colnames(female_med)[2] <- "Female_Median"
Sex_Diff <- merge(male_med, female_med, by = "Category")
Sex_Diff$Sex_Difference <- abs(Sex_Diff$Female_Median - Sex_Diff$Male_Median)
Sex_Diff$Category <- factor(Sex_Diff$Category, levels = c("Non-Social", "Social"))

# ── Shared theme ──────────────────────────────────────────────────────────────
shared_theme <- theme_minimal(base_size = 14) +
  theme(
    plot.title    = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, color = "gray40", size = 10)
  )

sex_colors <- c("Male" = "#FF8C00", "Female" = "#8A4D76")
cat_colors <- c("Social" = "#800080", "Non-Social" = "#FFA500")
ymax_raw   <- max(Bat_Medians$Median_Change, na.rm = TRUE)
# ── Plot 1: Interaction plot ──────────────────────────────────────────────────
p1 <- ggplot(Group_Stats,
             aes(x = Category, y = Median, color = Sex, group = Sex)) +
  geom_line(
    data  = Bat_Medians,
    aes(x = Category, y = Median_Change, group = Bat_ID, color = Sex),
    alpha = 0.3, linewidth = 0.6
  ) +
  geom_point(
    data  = Bat_Medians,
    aes(x = Category, y = Median_Change, color = Sex),
    alpha = 0.5, size = 2
  ) +
  geom_line(linewidth = 1.4) +
  geom_point(size = 4) +
  geom_errorbar(
    aes(ymin = Median - SE, ymax = Median + SE),
    width = 0.08, linewidth = 1
  ) +
  labs(
    title = "",
    x     = "Stimulus Category",
    y     = "Δ Heart Rate (BPM)",
    color = "Sex"
  ) +
  scale_color_manual(values = sex_colors) +
  shared_theme +
  theme(legend.position = "none") +
  coord_cartesian(clip = "off")

# ── Plot 2: Side-by-side boxplots ─────────────────────────────────────────────
p2 <- ggplot(Bat_Medians,
             aes(x = Category, y = Median_Change, fill = Sex)) +
  geom_boxplot(
    width         = 0.5,
    position      = position_dodge(width = 0.7),
    outlier.shape = NA,
    alpha         = 0.75
  ) +
  geom_point(
    aes(color = Sex),
    position = position_dodge(width = 0.7),
    size = 3, alpha = 0.9
  ) +
  stat_compare_means(
    aes(group = Sex),
    method       = "wilcox.test",
    label        = "p.signif",
    bracket.size = 0.8,
    label.y      = ymax_raw + 5
  ) +
  labs(
    title = "",
    x     = "Stimulus Category",
    y     = "Δ Heart Rate (BPM)",
    fill  = "Sex",
    color = "Sex"
  ) +
  scale_fill_manual(values  = sex_colors) +
  scale_color_manual(values = sex_colors) +
  shared_theme +
  theme(legend.position = "none") +
  coord_cartesian(clip = "off")
# ── Print & combine ───────────────────────────────────────────────────────────
print(p1)
print(p2)


combined <- ggarrange(p1, p2, ncol = 2, nrow = 1, labels = c("A", "B"))
print(combined)