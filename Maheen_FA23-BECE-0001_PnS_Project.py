##Maheen Farrukh Khan
##FA23-BECE-0001
##PnS Project
##pip install openpyxl
##pip install pandas numpy scipy matplotlib

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load data directly
data = pd.read_excel("Exam_Marks_SampleData.xlsx")
column = data.select_dtypes(include=np.number).columns[0]
x = data[column].dropna().values

def calculate_statistics():
    desc = {
        'Mean': np.mean(x),
        'Geometric Mean': stats.gmean(x),
        'Harmonic Mean': stats.hmean(x),
        'Median': np.median(x),
        'Mode': stats.mode(x, keepdims=True).mode[0],
        'Range': np.ptp(x),
        'Q1': np.percentile(x, 25),
        'Q2 (Median)': np.percentile(x, 50),
        'Q3': np.percentile(x, 75),
        'Quartile Deviation': (np.percentile(x, 75) - np.percentile(x, 25)) / 2,
        'Mean Deviation': np.mean(np.abs(x - np.mean(x))),
        'Variance': np.var(x, ddof=1),
        'Standard Deviation': np.std(x, ddof=1)
    }
    return pd.DataFrame(list(desc.items()), columns=['Measure', 'Value'])

def calculate_deciles_percentiles():
    deciles = {f'D{i}': np.percentile(x, i*10) for i in range(1, 10)}
    percentiles = {f'P{i}': np.percentile(x, i) for i in range(1, 100)}
    return pd.DataFrame(list({**deciles, **percentiles}.items()), columns=['Measure', 'Value'])

def calculate_coefficients():
    coeffs = {
        'Coeff. of Range': (np.max(x) - np.min(x)) / (np.max(x) + np.min(x)),
        'Coeff. of Quartile Deviation': (np.percentile(x, 75) - np.percentile(x, 25)) / (np.percentile(x, 75) + np.percentile(x, 25)),
        'Coeff. of Mean Deviation': np.mean(np.abs(x - np.mean(x))) / np.mean(x),
        'Coeff. of Variation': (np.std(x, ddof=1) / np.mean(x)) * 100
    }
    return pd.DataFrame(list(coeffs.items()), columns=['Measure', 'Value'])

def calculate_skewness_kurtosis():
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    skew_kurt = {
        "First Pearson's Skewness": 3 * (np.mean(x) - np.median(x)) / np.std(x, ddof=1),
        "Second Pearson's Skewness": 3 * (np.mean(x) - stats.mode(x, keepdims=True).mode[0]) / np.std(x, ddof=1),
        "Quartile Coeff. Skewness": (q3 + q1 - 2 * q2) / (q3 - q1),
        "Moment Coeff. Skewness": stats.skew(x),
        "Kurtosis": stats.kurtosis(x)
    }
    return pd.DataFrame(list(skew_kurt.items()), columns=['Measure', 'Value'])

def create_frequency_table():
    num_bins = int(1 + np.log2(len(x)))
    classes = pd.cut(x, bins=num_bins)
    freq_table = classes.value_counts().sort_index()
    class_mid = freq_table.index.map(lambda i: (i.left + i.right)/2)
    rel_freq = freq_table / freq_table.sum()
    cum_freq = freq_table.cumsum()
    return pd.DataFrame({
        'Class Interval': freq_table.index.astype(str),
        'Frequency': freq_table.values,
        'Mid-Value': class_mid.values,
        'Rel. Frequency': rel_freq.values,
        'Cum. Frequency': cum_freq.values
    })

def show_dataframe(df, title):
    win = tk.Toplevel()
    win.title(title)
    tree = ttk.Treeview(win, columns=list(df.columns), show='headings')
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor='center')
    for _, row in df.iterrows():
        tree.insert('', 'end', values=list(row))
    tree.pack(fill='both', expand=True)

def plot_graphs():
    freq_df = create_frequency_table()
    fig, axs = plt.subplots(3, 3, figsize=(18, 14))
    axs = axs.flatten()

    mid = freq_df['Mid-Value']
    freq = freq_df['Frequency']

    axs[0].bar(mid, freq, width=5)
    axs[0].set_title('Histogram')
    axs[0].tick_params(axis='x', rotation=45)

    axs[1].plot(mid, freq, marker='o')
    axs[1].set_title('Frequency Curve')
    axs[1].tick_params(axis='x', rotation=45)

    axs[2].plot(mid, freq, marker='o')
    axs[2].fill_between(mid, freq, alpha=0.3)
    axs[2].set_title('Frequency Polygon')
    axs[2].tick_params(axis='x', rotation=45)

    axs[3].plot(mid, freq_df['Cum. Frequency'], marker='o')
    axs[3].set_title('Cumulative Frequency Polygon')
    axs[3].tick_params(axis='x', rotation=45)

    axs[4].pie(freq, labels=None, autopct='%1.1f%%')
    axs[4].set_title('Pie Chart')
    axs[4].legend(freq_df['Class Interval'], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    axs[5].bar(freq_df['Class Interval'], freq, label='A')
    axs[5].bar(freq_df['Class Interval'], freq_df['Rel. Frequency'], label='B', alpha=0.6)
    axs[5].set_title('Multiple Bar Diagram')
    axs[5].legend()
    axs[5].tick_params(axis='x', rotation=45)

    axs[6].bar(freq_df['Class Interval'], freq, label='Total')
    axs[6].bar(freq_df['Class Interval'], freq_df['Rel. Frequency'], bottom=freq, label='Part', alpha=0.6)
    axs[6].set_title('Subdivided Bar Diagram')
    axs[6].legend()
    axs[6].tick_params(axis='x', rotation=45)

    axs[7].axis('off')
    axs[8].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.show()

def main_menu():
    root = tk.Tk()
    root.title("Statistics Project GUI")

    ttk.Button(root, text="1. Data Analysis (Ungrouped Data)", command=lambda: show_dataframe(calculate_statistics(), "Ungrouped Data Stats")).pack(pady=5)
    ttk.Button(root, text="2. Coefficients", command=lambda: show_dataframe(calculate_coefficients(), "Coefficient Measures")).pack(pady=5)
    ttk.Button(root, text="3. Skewness and Kurtosis", command=lambda: show_dataframe(calculate_skewness_kurtosis(), "Skewness and Kurtosis")).pack(pady=5)
    ttk.Button(root, text="4. Frequency Table (Grouped Data)", command=lambda: show_dataframe(create_frequency_table(), "Grouped Frequency Table")).pack(pady=5)
    ttk.Button(root, text="5. Tables (Graphs)", command=plot_graphs).pack(pady=5)
    ttk.Button(root, text="6. Exit", command=root.destroy).pack(pady=5)

    root.mainloop()

main_menu()
