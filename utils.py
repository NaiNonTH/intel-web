import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class MLDataEncoder:
    __data_encode = {}

    def __init__(self, df):
        for column in df.columns:
            self.__data_encode[column] = df[column].unique().tolist()

    def make_encoder(self, column_name):
        def data_encoder(value):
            return self.__data_encode[column_name].index(value)
        
        return data_encoder
        
    def encode(self, column_name, value):
        return self.__data_encode[column_name].index(value)
    
    def get_data_encode(self):
        return self.__data_encode

def get_scaler_and_encode(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["veil-type"])
    df = df[df["stalk-root"] != "?"]

    global_data_encoder = MLDataEncoder(df)
    data_encode = global_data_encoder.get_data_encode()

    for column in data_encode:
        data_encoder = global_data_encoder.make_encoder(column)
        df[column] = df[column].map(data_encoder)

    y = df["class"]
    x = df.drop(columns=["class"])

    scaler = MinMaxScaler().fit(x)

    return scaler, global_data_encoder

def kebab_to_heading(string: str):
    return " ".join([word.capitalize() for word in string.split("-")])
    

label_set = {
	"cap-shape": {
		"b": "bell",
		"c": "conical",
		"x": "convex",
		"f": "flat",
		"k": "knobbed",
		"s": "sunken"
	},
	"cap-surface": {
		"f": "fibrous",
		"g": "grooves",
		"y": "scaly",
		"s": "smooth"
	},
	"cap-color": {
		"n": "brown",
		"b": "buff",
		"c": "cinnamon",
		"g": "gray",
		"r": "green",
		"p": "pink",
		"u": "purple",
		"e": "red",
		"w": "white",
		"y": "yellow"
	},
	"bruises": {
		"t": "bruises",
		"f": "no"
	},
	"odor": {
		"a": "almond",
		"l": "anise",
		"c": "creosote",
		"y": "fishy",
		"f": "foul",
		"m": "musty",
		"n": "none",
		"p": "pungent",
		"s": "spicy"
	},
	"gill-attachment": {
		"a": "attached",
		"d": "descending",
		"f": "free",
		"n": "notched"
	},
	"gill-spacing": {
		"c": "close",
		"w": "crowded",
		"d": "distant"
	},
	"gill-size": {
		"b": "broad",
		"n": "narrow"
	},
	"gill-color": {
		"k": "black",
		"n": "brown",
		"b": "buff",
		"h": "chocolate",
		"g": "gray",
		"r": "green",
		"o": "orange",
		"p": "pink",
		"u": "purple",
		"e": "red",
		"w": "white",
		"y": "yellow"
	},
	"stalk-shape": {
		"e": "enlarging",
		"t": "tapering"
	},
	"stalk-root": {
		"b": "bulbous",
		"c": "club",
		"u": "cup",
		"e": "equal",
		"z": "rhizomorphs",
		"r": "rooted",
		"?": "missing"
	},
	"stalk-surface-above-ring": {
		"f": "fibrous",
		"y": "scaly",
		"k": "silky",
		"s": "smooth"
	},
	"stalk-surface-below-ring": {
		"f": "fibrous",
		"y": "scaly",
		"k": "silky",
		"s": "smooth"
	},
	"stalk-color-above-ring": {
		"n": "brown",
		"b": "buff",
		"c": "cinnamon",
		"g": "gray",
		"o": "orange",
		"p": "pink",
		"e": "red",
		"w": "white",
		"y": "yellow"
	},
	"stalk-color-below-ring": {
		"n": "brown",
		"b": "buff",
		"c": "cinnamon",
		"g": "gray",
		"o": "orange",
		"p": "pink",
		"e": "red",
		"w": "white",
		"y": "yellow"
	},
	"veil-type": {
		"p": "partial",
		"u": "universal"
	},
	"veil-color": {
		"n": "brown",
		"o": "orange",
		"w": "white",
		"y": "yellow"
	},
	"ring-number": {
		"n": "none",
		"o": "one",
		"t": "two"
	},
	"ring-type": {
		"c": "cobwebby",
		"e": "evanescent",
		"f": "flaring",
		"l": "large",
		"n": "none",
		"p": "pendant",
		"s": "sheathing",
		"z": "zone"
	},
	"spore-print-color": {
		"k": "black",
		"n": "brown",
		"b": "buff",
		"h": "chocolate",
		"r": "green",
		"o": "orange",
		"u": "purple",
		"w": "white",
		"y": "yellow"
	},
	"population": {
		"a": "abundant",
		"c": "clustered",
		"n": "numerous",
		"s": "scattered",
		"v": "several",
		"y": "solitary"
	},
	"habitat": {
		"g": "grasses",
		"l": "leaves",
		"m": "meadows",
		"p": "paths",
		"u": "urban",
		"w": "waste",
		"d": "woods"
	}
}