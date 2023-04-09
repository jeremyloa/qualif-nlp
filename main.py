from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import gutenberg, stopwords, wordnet
import random as rd
import string
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.probability import FreqDist
from nltk.classify import NaiveBayesClassifier, accuracy
import pickle 
import os 

def menu():
    print("\n\nJM-NLP")
    print("1. Input sentences (session 1-4)")
    print("2. Random sentence (NLTK Data) (session 1-4)")
    print("3. Input review (session 5)")
    print("4. Exit")
    chs = input(">> ")
    print("\n")
    return chs


def nltk_front(stn):
    # Tokenize
    sents = sent_tokenize(stn)
    words = word_tokenize(stn)

    # Stopwords
    eng = stopwords.words('english')
    removed = []
    for w in words:
        if w in eng or w in string.punctuation or w == '``' or w == "''" or w == "--":
            continue
        else:
            removed.append(w)

    # POS Tagging
    tagged = pos_tag(removed)

    # NER
    ner = ne_chunk(tagged)
    ner.draw()

    # Lemmatizing
    wml = WordNetLemmatizer()
    lmt = []
    for i, w in enumerate(removed):
        # when noun, change to adverb
        if tagged[i][1].startswith("N"):
            lmt.append(wml.lemmatize(w, pos="a"))
        # when adjective, change to noun
        elif tagged[i][1].startswith("JJ"):
            lmt.append(wml.lemmatize(w, pos="n"))
        # when adverb, change to adjective
        elif tagged[i][1].startswith("RB"):
            lmt.append(wml.lemmatize(w, pos="r"))
        # else, change to verb
        else:
            lmt.append(wml.lemmatize(w, pos="v"))

    # Frequency Distribution
    fd = FreqDist(lmt)
    for w, count in fd.most_common():
        print(f"{w}\n Count: {count}")
        # WordNet
        ss = wordnet.synsets(w)
        print("Definition:")
        for s in ss:
            print(f"- {s.definition()}")
            for l in s.lemmas():
                print(f"\tSynonym: {l.name()}")
                for a in l.antonyms():
                    print(f"\tAntonym: {l.name()}")

def train():
    bad = open("bad.txt", "r", encoding="utf-8").read()
    neutral = open("neutral.txt", "r", encoding="utf-8").read()
    good = open("good.txt", "r", encoding="utf-8").read()

    # Tokenize
    words = word_tokenize(bad) + word_tokenize(neutral) +  word_tokenize(good)  

    # Stopwords
    eng = stopwords.words('english')
    removed = []
    for w in words:
        if w in eng or w in string.punctuation or w == '``' or w == "''" or w == "--":
            continue
        else:
            removed.append(w)
    
    # Lemmatizing
    wml = WordNetLemmatizer()
    lmt = []
    for i, w in enumerate(removed):
        lmt.append(wml.lemmatize(w, pos="r"))

    # Labelling
    labeled = []
    for s in good.split("\n"):
        labeled.append((s, "good"))
    for s in neutral.split("\n"):
        labeled.append((s, "neutral"))
    for s in bad.split("\n"):
        labeled.append((s, "bad"))

    dataset = []
    for s, l in labeled:
        d = {}
        ws = word_tokenize(s)
        en = stopwords.words('english')
        r = []
        for w in ws:
            if w in en or w in string.punctuation or w == '``' or w == "''" or w == "--":
                continue
            else:
                r.append(w)
        lm = []
        for i, w in enumerate(r):
            lm.append(wml.lemmatize(w, pos="r"))
        for f in lmt:
            key = f 
            val = f in lm 
            d[key] = val 
        dataset.append((d, l))
    rd.shuffle(dataset)
    ctr = int(len(dataset) * 0.7)
    training = dataset[:ctr]
    testing = dataset[ctr:]
    classifier = NaiveBayesClassifier.train(training)
    # print(accuracy(classifier, training))
    file = open("model.pickle", "wb")
    pickle.dump(classifier, file)
    file.close()

if __name__ == '__main__':
    while True:
        choice = menu()

        if choice == "1":
            # sample sentence to input:
            # Jeremy's flight was yesterday at 5 AM. His flight is from Jakarta to Singapore. He loves to travel around the world.
            raw = input("Input sentences: ")
            nltk_front(raw)
        elif choice == "2":
            raw = list(gutenberg.sents("austen-sense.txt"))
            stns_raw = rd.sample(raw, 3)
            stns = " ".join([" ".join(stn) for stn in stns_raw])
            nltk_front(stns)
        elif choice == "3":
            if not os.path.exists("model.pickle"):
                train()
            
            file = open("model.pickle", "rb")
            classifier = pickle.load(file)
            file.close()

            review = input("Input review: ")
            wss = word_tokenize(review)
            res = classifier.classify(FreqDist(wss))
            print(res)

        else:
            break
