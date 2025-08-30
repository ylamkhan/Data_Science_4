import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt
import re


def main():
    df = pd.read_csv("../Train_knight.csv")
    df['knight'] = df['knight'].map({'Sith': 1, 'Jedi': 0})
    correlation_matrix = df.corr()
    plt.figure(figsize=(20, 18))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='viridis', linewidths=.5)
    plt.title('Correlation Matrix of Data Attributes')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()