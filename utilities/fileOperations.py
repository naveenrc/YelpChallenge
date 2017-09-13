import pandas as pd
import json


class FileOperations:
    @classmethod
    def getfile(cls, path):
        """
        Read json files
        """
        temp_list = []
        for line in open(path, 'r'):
            temp_list.append(json.loads(line))
        temp_df = pd.DataFrame(temp_list)
        return temp_df
