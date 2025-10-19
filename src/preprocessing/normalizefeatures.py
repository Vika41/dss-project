from sklearn.preprocessing import StandardScaler

def normalize_features(df, columns):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    return df_scaled
