import pandas as pd
from sklearn.model_selection import GroupKFold
path = r'C:/RVU/Sem5/student-complaint-intelligence/data/processed/task_candidate_urgency_severity.csv'
# Load and clean
df = pd.read_csv(path).dropna(subset=['text','severity']).reset_index(drop=True)
X = df['text'].astype(str).values
y = df['severity'].astype(str).values
groups = df['group_id'].values

gkf = GroupKFold(n_splits=5)
test_sizes = []
all_test = []
for train_idx, test_idx in gkf.split(X, y, groups):
    test_sizes.append(len(test_idx))
    all_test.extend(test_idx.tolist())

total = len(all_test)
unique = len(set(all_test))
missing = set(range(len(df))) - set(all_test)
print('Test fold sizes:', test_sizes)
print('Total test indices count:', total)
print('Unique test indices count:', unique)
print('Missing count:', len(missing))
print('Missing indices (first 20):', sorted(list(missing))[:20])
# Largest groups
group_counts = pd.Series(groups).value_counts()
print('Largest group sizes (top 10):', group_counts.head(10).to_dict())
# Null group IDs
null_groups = df['group_id'].isnull().sum()
print('Null group_id count:', null_groups)
