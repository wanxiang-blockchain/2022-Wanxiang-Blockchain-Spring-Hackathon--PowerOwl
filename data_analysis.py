import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re


def opcodePreprocess(opcode: str):
    if not opcode.startswith("opcode"):
        return re.sub(r'[0-9]', '', opcode)
    else:
        return None


def ByteCodeFiller(df: pd.DataFrame):
    copy_df = df.copy()
    for i in range(len(copy_df)):
        addr = copy_df["addr"][i]
        with open("./data/code/opcode/{}.txt".format(addr), 'r') as F:
            # Ignore first line
            for line in F.readlines()[1:]:
                operation: str = opcodePreprocess(line.split()[1])
                if operation != None:
                    copy_df.at[i, operation] = copy_df.at[i, operation] + 1
    copy_df.to_csv("./data/code/code_features.csv", index=False)


def ByteCodeReader(df: pd.DataFrame, common=15) -> dict:
    bytecodes = []
    addrs = df["addr"].tolist()
    for addr in addrs:
        bytecode = []
        with open("./data/code/opcode/{}.txt".format(addr), 'r') as F:
            # Ignore first line
            for line in F.readlines()[1:]:
                operation: str = opcodePreprocess(line.split()[1])
                if operation != None:
                    bytecode.append(operation)
        bytecodes.extend(bytecode)
    counter = Counter(bytecodes)
    mc = list(zip(*counter.most_common(common)))
    return {"op": np.array(mc[0]), "val": np.array(mc[1]) / sum(counter.values())}

if __name__ == '__main__':
    with open("dataset/code/opcode_list.txt",'r') as F:
        OPCODES = F.read().split()

    rawdf = pd.read_csv("ContractAddr.csv",sep='\t')
    rawdf[OPCODES] = 0


    df_ponzi = rawdf[rawdf["IS_PONZI"]==1]
    df_not_ponzi = rawdf[rawdf["IS_PONZI"]==0]

    df_ponzi.count(), df_not_ponzi.count()

    pdict = ByteCodeReader(df_ponzi)
    npdict = ByteCodeReader(df_not_ponzi)
    f, axes = plt.subplots(1, 2, figsize=(25, 12))
    sns.barplot(x="op", y="val", data=pdict, ax=axes[0]).set_title('ponzi')
    sns.barplot(x="op", y="val", data=npdict, ax=axes[1]).set_title('non-ponzi')

    ByteCodeFiller(rawdf)

    df = pd.read_csv("data/code/code_features.csv", )
    df.drop(df.columns[[0]], axis=1, inplace=True)

    # 80-1 columns
    fig, axes = plt.subplots(ncols=10, nrows=8, figsize=(60, 48))
    for column, ax in zip(df.columns, axes.flat):
        if (column == "IS_PONZI"): continue
        sns.histplot(
            df, x=column, hue="IS_PONZI", element="step", stat="density", common_norm=False, ax=ax, )
    fig.savefig("./image/opcode_ponzi_compare.jpg")

    # Correlation plot

    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20, 16))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    f.savefig("./image/corr.jpg")

