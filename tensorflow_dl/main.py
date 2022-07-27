## Example python file using tensorflow som module with modified dtw
from tf_som_dtw import SOM

if __name__ == "__main__":
    print("Hello")

    som = SOM(6, 6, 4, 0.5, 0.5, 100)
