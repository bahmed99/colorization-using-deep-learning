import os
from importlib.machinery import SourceFileLoader

def listModels():
    models = []
    for root, dirs, files in os.walk('models'):
        # Only process subdirectories and not the root 'models' directory
        if root != 'models':
            # Traverse all files in the current directory
            for file in files:
                # files that start with 'Colorize' and have a '.py' extension
                if file.startswith('Colorizer') and file.endswith('.py'):
                    model_path = os.path.join(root, file)
                    # Extract the model name from the file name
                    model_name = os.path.splitext(file)[0]
                    # Remove the 'Colorize' prefix from the model name
                    model_name = model_name.replace('Colorizer', '')
                    models.append(model_name)
    return models

def getColorizerClassFromName(model):
    className = "Colorizer"+model
    target = className+".py"
    for root, _, files in os.walk('models'):
        # Only process subdirectories and not the root 'models' directory
        if root != 'models' and 'models' in root:
            if target in files:
                module = SourceFileLoader(className, os.path.join(root, target)).load_module()
                return getattr(module, className)