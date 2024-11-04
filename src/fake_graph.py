import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme()

# For the image quality of the graphic.
sns.set(rc={"figure.dpi":300})
# For the size of the graphics
sns.set(rc = {"figure.figsize":(6,3)})


models = {
    'Qwen2.5-7B \n(Teacher Model)': {
        'IFEval': 75.85,
        'BBH': 34.89,
        'MMLU-PRO': 36.52
    },
    'Qwen2.5-1.5B \n(Student Model Distilled)':
        {
            'IFEval': 71.47,
            'BBH': 27.16,
            'MMLU-PRO': 27.83

        },
    'Qwen2.5-1.5B \n(Student Model Pretrained)':
        {
            'IFEval': 33.71,
            'BBH': 13.7,
            'MMLU-PRO': 16.68
        }
}

# df = pd.DataFrame.from_dict(models, orient='index')
# df.reset_index(inplace=True, names='model')
#
# print(df.head())
#
# test = pd.read_csv("~/Downloads/penguins.csv")
#
# print(test.head())

data = pd.DataFrame(models).reset_index().melt(id_vars='index', var_name='Model', value_name='Value')
data.rename(columns={'index': 'Dataset'}, inplace=True)

data_avg = data.groupby("Model")["Value"].mean().reset_index()
data_avg.rename(columns={"Value": "Average Score"}, inplace=True)

# Add average performance scores to the original data for plotting
data_with_avg = pd.concat([data, pd.DataFrame({
    'Dataset': ['Average'] * len(data_avg),
    'Model': data_avg['Model'],
    'Value': data_avg['Average Score']
})])

data = data_with_avg
# Set up the plot
# sns.set(style="whitegrid")
plt.figure(figsize=(15, 5))

# Draw the barplot for each model
ax = sns.barplot(data=data, x="Dataset", y="Value", hue="Model", errorbar=None)


# Add labels and title
plt.xlabel("Dataset")
plt.ylabel("Score")
plt.title("Performance of Different Models on Various Datasets")
plt.legend(title="Model")
# plt.show()

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", padding=2)


# ax = sns.barplot(df, y='IFEval', x='model')
# fig = ax.get_figure()
plt.savefig('graph.png')
