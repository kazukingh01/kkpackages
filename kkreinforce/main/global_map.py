import pandas as pd
import folium


if __name__ == "__main__":
    df = pd.read_csv("../data/s59h30megacities_utf8.csv", sep="\t")
    m = folium.Map(location=[45.5236, -122.6750])
    m.save('index.html')

