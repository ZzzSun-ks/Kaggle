import dataset

def process(filename):
    df = dataset.load_data(filename)

    df = df.loc[:, df.std() > 1e-5]

    return df