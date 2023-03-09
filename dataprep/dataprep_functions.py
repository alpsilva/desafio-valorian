from sklearn.model_selection import train_test_split
import pandas

def create_time_window_data(df: pandas.DataFrame, interval_in_seconds: int = 3, overlap: float = 0.75):
    """
    Receives a Pandas Dataframe, an interval in seconds (default 3) and an overlap value, ranging from 0 to 1 (default 0.75).
    Generates and returns a new Pandas Dataframe using the shifting window data augmentation technique, using the timestamp of
    the original df as the axis and the interval and overlap given as parameters for the window and step sizes.
    """

    time_interval = interval_in_seconds * 1000000000

    step_size = time_interval * (1 - overlap)
    window_start = df["timestamp"].min()
    window_end = window_start + time_interval
    end = df["timestamp"].max()

    data = []
    while window_end < end:
        window = df[(df["timestamp"] >= window_start) & (df["timestamp"] < window_end)]

        if len(window) > 0:
            data.append(window)
        window_start += step_size
        window_end = window_start + time_interval
    
    new_df = pandas.concat(data)
    return new_df

def process_dataset(file_path: str):
    """
    Receives the filepath of souce dataset (the dataset must follow this exact structure).
    Loads the dataset into a Pandas Dataframe, removes null and noise rows,
    augments the available data with a time window of 3 seconds and 75% overlap
    and finally splits the dataset into and returns:
    - training set (70%);
    - testing set (20%);
    - validation set (10%).
    """
    columns = ["individuo", "atividade", "timestamp", "x_accel", "y_accel", "z_accel"]
    
    df = pandas.read_csv(
        file_path,
        names=columns,
        header=None,
        sep=',',
        lineterminator=';'
    )

    df = df.dropna()
    df = df.loc[(df["timestamp"] != 0) & (df["x"] != 0) & (df["y"] != 0) & (df["z"] != 0)]

    time_interval = 3
    overlap = 0.75
    df = create_time_window_data(df, time_interval, overlap)

    features = ["x_accel", "y_accel", "z_accel"]
    label = ["atividade"]

    X = df[features]
    y = df[label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.125, random_state = 42)

    return X_train, y_train, X_test, y_test, X_valid, y_valid


