#%%
import pandas as pd
import numpy as np
import numba as nb


class Origin:
    def __init__(self, Top, Bottom, CrosswebCenter) -> None:
        self.Top = Top
        self.Bottom = Bottom
        self.CrosswebCenter = CrosswebCenter


def maskTopBottom(Top, Bottom, origin, downwebOffset):
    maskTop = Top > origin.Bottom - downwebOffset
    maskTop = np.logical_and(maskTop, Top < origin.Bottom + downwebOffset)

    maskBottom = Bottom > origin.Top - downwebOffset
    maskBottom = np.logical_and(maskBottom,
                                Bottom < origin.Top + downwebOffset)
    return np.logical_or(maskTop, maskBottom)


def maskCrosswebCenter(CrosswebCenter, origin: Origin, crosswebOffset):
    maskLeft = CrosswebCenter > origin.CrosswebCenter - crosswebOffset
    maskRight = CrosswebCenter < origin.CrosswebCenter + crosswebOffset
    return np.logical_and(maskLeft, maskRight)


def maskBbox(Top, Bottom, CrosswebCenter, origin, downwebOffset,
             crosswebOffset):
    TopBottom = maskTopBottom(Top, Bottom, origin, downwebOffset)
    LeftRight = maskCrosswebCenter(CrosswebCenter, origin, crosswebOffset)
    return np.logical_and(TopBottom, LeftRight)


df = pd.read_excel(path)
df["Label"] = 0
downwebOffset = 2
crosswebOffset = 0.1

maskScam = df.Type == "Type"
df = df.loc[maskScam]

maskZero = df.Label.values == 0
label = 0

while (maskZero).any():
    label += 1
    df_unlabel = pd.DataFrame(df.loc[maskZero])
    masklabeled = df_unlabel.iloc[:, -1] == label

    df_unlabel.iloc[0, -1] = label

    maskEnlarged = masklabeled != (df_unlabel.iloc[:, -1] == label)
    i = 0
    while maskEnlarged.any():
        df_labeled = df_unlabel.loc[df_unlabel.Label == label]
        masklabeled = df_unlabel.iloc[:, -1] == label
        for idx in range(df_labeled.shape[0]):
            row = df_labeled.iloc[idx]
            origin = Origin(*[row.Top, row.Bottom, row.CrosswebCenter])

            Top = df_unlabel.Top.values
            Bottom = df_unlabel.Bottom.values
            CrosswebCenter = df_unlabel.CrosswebCenter.values

            mask = maskBbox(Top, Bottom, CrosswebCenter, origin, downwebOffset,
                            crosswebOffset)

            df_unlabel.loc[mask, "Label"] = label

        maskEnlarged = masklabeled != (df_unlabel.iloc[:, -1] == label)
    df.loc[maskZero, "Label"] = df_unlabel.Label.values
    maskZero = df.Label.values == 0
df
