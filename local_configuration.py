"""Make the local configuration available."""


import os
import json
import pandas as pd


class LocalConfiguration(object):
    """Make available the local configutation for the machine learning labs."""

    def __init__(self):
        """Initialize the environment."""
        self.configurationFile = ".env"
        self.hasConfiguration = self.configuration_exist()
        self.configuration = None

        try:
            # Try to load configuration.
            self.getConfiguration()
        except:
            pass

    def __repr__(self):
        """Create a configuration."""
        line = "Has configuration: {}".format(self.hasConfiguration)
        line += "\nConfiguration file: {}".format(self.configurationFile)
        return line

    def configuration_exist(self):
        """Find out if local file conf is present."""
        return os.path.isfile(self.configurationFile)

    def getConfiguration(self):
        """Get configuration from the file."""
        if not self.configuration_exist():
            # TODO: raise more specific exception
            raise Exception

        with open(self.configurationFile) as file:
            configuration = json.load(file)

        self.configuration = configuration

    def data_file_exists(self):
        """Confirm the existanse of the data file at the user's machine."""
        pass

    def read_data(self):
        """Get the data from any resource."""
        if self.hasConfiguration and self.configuration['file_path'] is not None:
            data = pd.read_csv(self.configuration['file_path'],
                               header=None).values
        else:
            data = pd.read_csv(url_data, header=None).values

        return data


if __name__ == "__main__":
    a = LocalConfiguration()
    print (a.read_data())
