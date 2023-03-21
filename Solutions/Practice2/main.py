import nltk
import string
from nltk.corpus import brown
from nltk.probability import *
from ngrams import getBestBigrams, getBestTrigrams
from normalization import MyNormalizer
from util import downloadDataset, getWordsInCategories, getMostCommonWordsSet

if __name__ == "__main__":

    downloadDataset("brown")

    # Print all the categories in "brown"
    print(f"Categorie dataset: {brown.categories()}")
    categorie_eccetto_adventure = brown.categories().remove("adventure")


    # Estrazione parole
    parole_adventure = getWordsInCategories(brown, ["adventure"])
    parole_non_adventure = getWordsInCategories(brown, categorie_eccetto_adventure)

    # Normalizzazione
    normalizer = MyNormalizer()
    parole_adventure_normalizzate = normalizer.normalizeWords(parole_adventure, remove_stopwords=True)
    parole_non_adventure_normalizzate = normalizer.normalizeWords(parole_non_adventure, remove_stopwords=True)

    print(f"Dimensione vocabolario normalizzato per categoria ADVENTURE: {len(parole_adventure_normalizzate)}")
    print(f"Dimensione vocabolario normalizzato per tutte le altre categorie: {len(parole_non_adventure_normalizzate)}")


    # Calcolo numero di occorrenze
    parole_piu_frequenti_adventure = FreqDist(parole_adventure_normalizzate)
    parole_piu_frequenti_non_adventure = FreqDist(parole_non_adventure_normalizzate)

    print("\nStampo le prime 10 parole più comuni per ADVENTURE e il complementare")
    print(parole_piu_frequenti_adventure.most_common(10))
    print(parole_piu_frequenti_non_adventure.most_common(10))


    # Intersezione tra le prime 100 parole più frequenti
    # ogni elemento in parole_piu_frequenti_adventure.most_common(100) è una coppia (parola, occorrenze)
    # usiamo la list comprehension per estrarre le parole (con x[0]) da ognuno di questi elementi
    # trasformiamo la lista in set per avere le operazioni insiemistiche
    a = set([ x[0] for x in parole_piu_frequenti_adventure.most_common(100) ])
    b = set([ x[0] for x in parole_piu_frequenti_non_adventure.most_common(100) ])
    intersezione_100_parole_piu_frequenti = a.intersection(b)
    print("Interzeione 100 parole più frequenti:")
    print(intersezione_100_parole_piu_frequenti)
    print("\n\n")


    # Bigrammi
    print("Calcolo bigrammi...")
    migliori_bigrammi_adventure = getBestBigrams(parole_adventure_normalizzate, normalizer.punctuation, number=20)
    print(f" # Migliori bigrammi in ADVENTURE: {migliori_bigrammi_adventure}")

    migliori_bigrammi_non_adventure = getBestBigrams(parole_non_adventure_normalizzate, normalizer.punctuation, number=20)
    print(f" # Migliori bigrammi nelle altre categorie: {migliori_bigrammi_non_adventure}")
    print("\n\n")




    print("Calcolo trigrammi...")

    migliori_trigrammi_adventure = getBestTrigrams(parole_adventure_normalizzate, normalizer.punctuation, number=10)
    print(f" # Migliori trigrammi in ADVENTURE: {migliori_trigrammi_adventure}")
    migliori_trigrammi_non_adventure = getBestTrigrams(parole_non_adventure_normalizzate, normalizer.punctuation, number=10)
    print(f" # Migliori trigrammi nelle altre categorie: {migliori_trigrammi_non_adventure}")











