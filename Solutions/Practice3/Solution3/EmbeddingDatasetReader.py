import sqlite3
import sys

import numpy as np


class EmbeddingDatasetReader:

    def __init__(self, sqlite_path):
        """
        Read sqlite embeddings from sqllite_path and returns them into a pandas DataFrame
        """

        print(f"Reading embeddings from file '{sqlite_path}'...")

        """
        An index has been created on the database:
        
        CREATE INDEX "key_index" ON "store" (
            "key"	ASC
        );
        """
        self.con = sqlite3.connect(sqlite_path)

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
        try:
            cursor = self.con.cursor()
            cursor.execute('SELECT * FROM store WHERE key=? LIMIT 1', (word, ))
            if cursor.rowcount != 0:
                data = cursor.fetchone()
                cursor.close()
                return data
            else:
                print(
                    f"Word '{word}' not found in embedding database. Consider looking for a similar word using Minimum Edit Distance",
                    file=sys.stderr)
                return None
        except(sqlite3.OperationalError):
            return self.getWordEntry(word)


    def getWordEmbedding(self, word: str) -> np.array:
        entry = self.getWordEntry(word)
        if entry == None:
            return np.zeros(128)
        else:
            return np.array(EmbeddingDatasetReader.getEmbeddingVectorFromEntry(entry))


