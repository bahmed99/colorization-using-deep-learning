from models import Colorizer


class ColorizerReference(Colorizer.Colorizer):
    def __init__(self, inputFolderPath, colorizationName, hyperParameters) -> None:
        super().__init__(inputFolderPath, colorizationName, hyperParameters)
