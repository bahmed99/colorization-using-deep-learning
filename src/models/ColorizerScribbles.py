from models import Colorizer


class ColorizerScribbles(Colorizer.Colorizer):
    def __init__(self, inputFolderPath, colorizationName, hyperParameters) -> None:
        """
        hyperParameters : dictionary
          "scribbles" : scribbles path as string
        """
        super().__init__(inputFolderPath, colorizationName, hyperParameters)
