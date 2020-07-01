import pandas as pd
import numpy as np
import cv2
from typing import List

from folium import MacroElement
from folium.features import Template
class DivIcon(MacroElement):
    def __init__(self, html='', size=(30,30), anchor=(0,0), style=''):
        """TODO : docstring here"""
        super(DivIcon, self).__init__()
        self._name = 'DivIcon'
        self.size = size
        self.anchor = anchor
        self.html = html
        self.style = style

        self._template = Template(u"""
            {% macro header(this, kwargs) %}
              <style>
                .{{this.get_name()}} {
                    {{this.style}}
                    }
              </style>
            {% endmacro %}
            {% macro script(this, kwargs) %}
                var {{this.get_name()}} = L.divIcon({
                    className: '{{this.get_name()}}',
                    iconSize: [{{ this.size[0] }},{{ this.size[1] }}],
                    iconAnchor: [{{ this.anchor[0] }},{{ this.anchor[1] }}],
                    html : "{{this.html}}",
                    });
                {{this._parent.get_name()}}.setIcon({{this.get_name()}});
            {% endmacro %}
            """)


class WorldMap(object):
    """
    WorldMap を緯度と経度から描画して、それに対して線を引いたりする
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["lat"] = (-1 * (self.df["lat"] - 90.0 ) / 0.5).astype(int)
        self.df["lon"] = (     (self.df["lon"] + 180.0) / 0.5).astype(int)
    
    def initialize(self) -> np.ndarray:
        img = np.zeros((180*2,360*2, 3)).astype(np.uint8)
        return self.drawcities(img)
    
    def drawcities(self, img: np.ndarray):
        img = img.copy()
        for lat, lon in self.df[["lat", "lon"]].values:
            img[lat, lon, :] = (255, 0, 0)
        return img
    
    def drawlocation(self, img: np.ndarray, city: str, value: (int, int, int)) -> np.ndarray:
        img = img.copy()
        lat, lon = self.df[self.df["capital_en"] == city].iloc[0][["lat", "lon"]].values
        img[lat, lon, :] = value
        return img
    
    def drawline(self, img: np.ndarray, city_from: str, city_to: str) -> np.ndarray:
        img = img.copy()
        lat_from, lon_from = self.df[self.df["capital_en"] == city_from].iloc[0][["lat", "lon"]].values
        lat_to,   lon_to   = self.df[self.df["capital_en"] == city_to  ].iloc[0][["lat", "lon"]].values
        # point は x, y で入力しないといけないので、img の順番と逆にする必要がある
        img = cv2.line(img, (int(lon_from), int(lat_from)), (int(lon_to), int(lat_to)), (0,255,0), 1)
        return img

    def list_capital(self) -> List[str]:
        return self.df["capital_en"].tolist()
    
    @classmethod
    def conv_to_torch(cls, img: np.ndarray) -> np.ndarray:
        return np.array([img[:, :, i] for i in range(img.shape[-1])])
    
    @classmethod
    def conv_from_torch(cls, img: np.ndarray) -> np.ndarray:
        return cv2.flip(np.rot90(img.T, 3), 1)
    
    def show(self, img: np.ndarray, ):
        cv2.imshow("test", img)
        cv2.waitKey(0)



class TSPModelBase(object):
    def __init__(self, file_csv: str="../data/s59h30megacities_utf8.csv", n_capital: int=None):
        df = pd.read_csv(file_csv, sep="\t")
        df = df[df["iscapital"] == 1]
        df["capital_en"] = df["capital_en"].replace(r"\s", "_", regex=True)
        if n_capital is not None:
            ndf = np.random.permutation(df["capital_en"].unique())[:n_capital] # 都市を限定する
            df  = df[df["capital_en"].isin(ndf)]
        self.df = df.copy()
        self.wmp = WorldMap(self.df)

    def distance(self, city1: str, city2: str) -> float:
        """
        city1 から city2 への距離を計算する
        """
        if city1 == city2:
            val = 0
        else:
            df = self.df
            se1 = df[df["capital_en"] == city1].iloc[0]
            se2 = df[df["capital_en"] == city2].iloc[0]
            val = self.cal_rho(se1["lon"], se1["lat"], se2["lon"], se2["lat"]) / 1000.
        return val
    
    def get_lat_lon(self, city: str) -> (float, float, ):
        df = self.df
        return df[df["capital_en"] == city].iloc[0][["lat", "lon"]].values

    @classmethod
    def cal_rho(cls, lon_a,lat_a,lon_b,lat_b):
        ra=6378.140  # equatorial radius (km)
        rb=6356.755  # polar radius (km)
        F=(ra-rb)/ra # flattening of the earth
        rad_lat_a=np.radians(lat_a)
        rad_lon_a=np.radians(lon_a)
        rad_lat_b=np.radians(lat_b)
        rad_lon_b=np.radians(lon_b)
        pa=np.arctan(rb/ra*np.tan(rad_lat_a))
        pb=np.arctan(rb/ra*np.tan(rad_lat_b))
        xx=np.arccos(np.sin(pa)*np.sin(pb)+np.cos(pa)*np.cos(pb)*np.cos(rad_lon_a-rad_lon_b))
        c1=(np.sin(xx)-xx)*(np.sin(pa)+np.sin(pb))**2/np.cos(xx/2)**2
        c2=(np.sin(xx)+xx)*(np.sin(pa)-np.sin(pb))**2/np.sin(xx/2)**2
        dr=F/8*(c1-c2)
        rho=ra*(xx+dr)
        return float(rho)

