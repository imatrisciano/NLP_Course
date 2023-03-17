import nltk
import string
from nltk.corpus import brown
from nltk.probability import *


def GetBestBigrams(words, punctuations, number):
    bigram_measures = nltk.collocations.BigramAssocMeasures
    bigrams_finder = nltk.collocations.BigramCollocationFinder.from_words(words)
    bigrams_finder.apply_word_filter(lambda w: w.lower() in punctuations)
    return bigrams_finder.nbest(bigram_measures.pmi, number)

def GetBestTrigrams(words, punctuations, number):
    trigram_measures = nltk.collocations.TrigramAssocMeasures
    trigrams_finder = nltk.collocations.TrigramCollocationFinder.from_words(words)
    trigrams_finder.apply_word_filter(lambda w: w.lower() in punctuations)
    return trigrams_finder.nbest(trigram_measures.pmi, number)


if __name__ == "__main__":

    # Download "brown" dataset if needed
    try:
        nltk.data.find("brown")
    except LookupError:
        nltk.download("brown")

    # Print all the categories in "brown"
    print(f"Categorie dataset: {brown.categories()}")
    categorie_eccetto_adventure = brown.categories().remove("adventure")

    fileids_adventure = brown.fileids(categories="adventure")
    fileids_non_adventure = brown.fileids(categories=categorie_eccetto_adventure)


    # Estrazione parole
    # Qui non usiamo i set altrimenti si eliminano i duplicati e il conteggio di ogni parola sarà 1
    parole_adventure = brown.words(fileids_adventure)
    parole_non_adventure = brown.words(fileids_non_adventure)

    # Eliminazione punteggiatura
    punteggiatura = set(string.punctuation)
    punteggiatura.add('``')
    punteggiatura.add('\'\'')

    parole_adventure_filtrate = [x for x in parole_adventure if x not in punteggiatura]
    parole_non_adventure_filtrate = [x for x in parole_non_adventure if x not in punteggiatura]

    print(f"Dimensione vocabolario filtrato, ma non normalizzato per categoria ADVENTURE: {len(parole_adventure_filtrate)}")
    print(f"Dimensione vocabolario filtrato, ma non normalizzato per categoria tutte le altre categorie: {len(parole_non_adventure_filtrate)}")


    # Calcolo numero di occorrenze
    parole_piu_frequenti_adventure = FreqDist(parole_adventure_filtrate)
    parole_piu_frequenti_non_adventure = FreqDist(parole_non_adventure_filtrate)

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
    print("Calcolo bigrammi")

    migliori_bigrammi_adventure = GetBestBigrams(parole_adventure_filtrate, punteggiatura, number=20)
    print(f"Migliori bigrammi in ADVENTURE: {migliori_bigrammi_adventure}")

    migliori_bigrammi_non_adventure = GetBestBigrams(parole_non_adventure_filtrate, punteggiatura, number=20)
    print(f"Migliori bigrammi nelle altre categorie: {migliori_bigrammi_non_adventure}")




    print("\n\n")
    print("Calcolo trigrammi")

    migliori_trigrammi_adventure = GetBestTrigrams(parole_adventure_filtrate, punteggiatura, number=10)
    print(f"Migliori trigrammi in ADVENTURE: {migliori_trigrammi_adventure}")
    migliori_trigrammi_non_adventure = GetBestTrigrams(parole_non_adventure_filtrate, punteggiatura, number=10)
    print(f"Migliori trigrammi nelle altre categorie: {migliori_trigrammi_non_adventure}")










