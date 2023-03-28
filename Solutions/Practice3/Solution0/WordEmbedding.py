import sqlite3
import sys

import numpy as np


class WordEmbedding:
    def __init__(self, sqlite_path):
        """
        Read sqlite embeddings from sqllite_path and returns them into a pandas DataFrame
        """

        print(f"Reading embeddings from file '{sqlite_path}'...")

        con = sqlite3.connect(sqlite_path)

        # data = pd.read_sql_query("SELECT * FROM store", con)
        # we decided not to use the DataFrame structure due to excessive memory overhead

        cursor = con.cursor()
        cursor.execute('SELECT * FROM store')
        data = cursor.fetchall()
        con.close()

        self.dataset = data

    @staticmethod
    def getWordFromEntry(entry) -> str:
        return entry[0]

    @staticmethod
    def getEmbeddingVectorFromEntry(entry) -> list:
        return entry[1:129]

    @staticmethod
    def getIdFromEntry(entry) -> id:
        return entry[-1]

    def getWordEntry(self, word: str) -> tuple:
        for entry in self.dataset:
            if (entry[0] == word):
                return entry
        print(
            f"Word '{word}' not found in embedding database. Consider looking for a similar word using Minimum Edit Distance",
            file=sys.stderr)

        return None

    def getWordEmbedding(self, word: str) -> np.array:
        for entry in self.dataset:
            if (entry[0] == word):
                return np.array(WordEmbedding.getEmbeddingVectorFromEntry(entry))

        print(
            f"Word '{word}' not found in embedding database. Consider looking for a similar word using Minimum Edit Distance",
            file=sys.stderr)

        return np.zeros(128)
