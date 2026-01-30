# %%
import pandas as pd
import time
import os
# %%
results_log_path = "../results/beta_results/written_feedback.txt"
os.makedirs(os.path.dirname(results_log_path), exist_ok=True)

ts = time.strftime("%Y%m%d")
results_log = open(results_log_path, "a")  # append mode
results_log.write(f"\n=== Run {ts} ===\n")

df = pd.read_csv("../data/beta_data/feedback_written_jan26.csv")
df.head()
# %%
df.columns = ['timestamp', 'clarity_project_goals',
       'sensical_instruction',
       'difficulty',
       'FT_what_difficult',
       'useful_tutorial',
       'useful_task_helptext',
       'useful_field_guide',
       'did_you_read_additional_info',
       'useful_additional_info',
       'FT_additional_info_suggestions',
       'suitability_talk_boards',
       'FT_talkboards_requests',
       'useful_project_name',
       'FT_name_suggestions',
       'suitable_zooniverse_platform',
       'FT_platform_suggestions',
       'FT_comments_general',
       'further_interest']
df.head()
# %%

cat_cols = [x for x in df.columns if not x.startswith("FT_") and x != "timestamp"]

# get value counts for each categorical column in percentages
for col in cat_cols:
    print(f"## {col} ##")
    print(df[col].value_counts(normalize=True) * 100)
    print("\n")
    results_log.write(f"## {col} ##\n")
    results_log.write(str(df[col].value_counts(normalize=True) * 100))
    results_log.write("\n\n")

# close the log file
results_log.close()

# %%
