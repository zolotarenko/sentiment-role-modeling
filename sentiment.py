# coding=utf-8

################################################
############## Projekt Informationen ###########
################################################
# Seminar: Manfred Stede, Sentiment Analyse (2018)
# Author: Olha Zolotarenko
# Matr.Nr: 787894
# Thema: Role Modeling
# Python Version: Python 3
# To be compiled with python 3
# Quellen:
# German Sentiment Lexicon - http://bics.sentimental.li/files/8614/2462/8150/german.lex
# https://liris.cnrs.fr/Documents/Liris-6508.pdf

##################################
############ Prinzip #############
##################################
# Ziel: Relations und Attitude des Autors in einem Satz berechnen
# Die Implementierung basiert auf der syntaktischer Struktur des Satzes.
# Der Syntaxbaum wird als OrderedDict gespeichert, in dem zu jedem Verb
# bottom-up Tochterknoten gesucht werden.
# So wird Attitude des Autors und Relations zwischen den Entitäten aus dem Wert
# des Verb und jeder seiner Töchter berechnet. Tritt in einem Satzteil Negation auf,
# so wird der Wert des Satzteils umgekehrt.

# Bemerkungen (Quellen von möglichen Fehler in der Berechnung):
# 1. child.lemma_ 'Sieg' --> 'siegen'. Im Lexikon wird 'siegen' gesucht und falls nicht da - 0 zurückgegeben,
# was zu Ungenauigkeiten führt.
# 'Sieg' in self.lexicon --> False
# 2. Nicht alle Entitäten werden vom Spacy korrekt erfasst
# z.B "Frankreich profitiert von der Niederlage der Engländer"  ==> "Niederlage der Engländer" wird als Entität erfasst
# 3. Die trennbaren Verben wie z.B. "vorwerfen" sind von SpaCy nicht als solche erfasst (die Präfixe gehen verlosen).
# Daher führt das zu falschen Berechnungen eines Satzes.
# z.B. in dem Satz "Deutschland und Frankreich werfen Grossbritanien vor,
# dass es die Beziehungen zu den USA nicht bevorzugen will ."
# 4. Die Sätze wie 'Das Spiel macht dem Minister grossen Spass .' werden ebenso nicht korekt berechnet,
# da die Entitäten bei SpaCy falsch bestimmt werden: Entities: [Minister grossen Spass]!
# 5. Pronomen werden auch nicht als Entitäten erkannt!
# Der Datz 'Stefan kritisiert , dass Marie bedauert , den Peter belogen zu haben .' wird korrekt
# analysiert, aber 'Er kritisiert , dass sie bedauert , den Peter belogen zu haben .' nicht mehr.

import time
import json
import spacy
from collections import OrderedDict

###############################################
#### Klasse zur Sentimentanalyse der Sätze ####
###############################################

# Klasse zur automatischen Rollen-Erfassung (Attitude und Beziehungen zwischen den Entitäten)
# Attitüde ist 1 wenn positiv, -1 wenn negativ und 0 wenn neutral
# Beziehungen sind in der Form (source, sentiment, target) mit sentiment = 1, -1 oder 0

class Role_Modeling:

    def __init__(self, nlp_root):

        print("-%-%-%-%-%-%-%-%-%-%-%- INPUT -%-%-%-%-%-%-%-%-%-%-%-")

        # Lexikon abrufen
        with open('result.json') as f:
            self.lexicon = json.load(f)

        # Neue Wörter dem Lexikon hinzufügen
        self.lexicon['verlieren'] = -1
        self.lexicon['Niederlage'] = -1
        self.lexicon['Misserfolg'] = -1
        self.lexicon['siegen'] = 1
        self.lexicon['erfolgen'] = 1
        self.lexicon['erfolgen'] = 1
        self.lexicon['vorwerfen'] = -1

        # Tree class instance
        struct = Tree(sent)

        # Lists of values for attirudes
        self.total_val_attitudes = []

        # Tree repräsentiert als Ordered Dictionary in Form:
        # 'verb', [liste von abhängigen Knoten]
        self.tree = struct.check_noun_phrase()
        print("Tree: " + str(self.tree))

        # Speichern der Wurzel
        self.root = nlp_root
        self.sb = struct.sb
        # print(self.sb)

        # Variable zum Umkehren der Sentimente
        self.rev_sentiments = False

        # Subkategorisierungslisten
        self.subcat = [child.dep_ for child in self.root.children]
        self.subcat_children = []
        self.root_deps = []
        print(self.subcat)

        # Detected entities in the sentence
        self.reversed_entities = list(reversed(struct.ents))
        print('Entities: ' + str(self.reversed_entities))

        # Einstellungen des Autors
        self.attitudes = dict()

        # Relationen zwischen den Entitäten
        self.relations = set()

        # Already reversed
        self.total_val_attitudes = []
        self.total_val_relations = dict()
        self.ent_pairs = self.pairs(self.reversed_entities)

        # Analyse des Satzes wird hier gestartet
        self.analyze()

    ##############################################
    #### Methode zur syntaxabhängigen Analyse ####
    ######## der Senimente in einem Satz #########
    ##############################################

    def analyze(self):

        # Hier erfolgt Iteration durch die Baumstruktur
        # bottom-up (reversed)
        for node, children in reversed(self.tree.items()):

            word = nlp(node)
            token = word[0].lemma_

            # Die Werte der Wörter werden in der Struktur des Baums
            # analysiert. Exeption wird benötigt, um KeyError zu vermeiden,
            # da die Werte in dem Dictionary gespeichet sind.
            try:

                print(token, self.get_val(token))
                self.root_value = int(self.get_val(token))
                self.root_deps = [self.root_value]
                self.total_val_attitudes.append(self.root_value)

                for child in children:

                    self.subcat_children = [child.dep_ for child in children]

                    print(child, self.get_val(child.lemma_))
                    if self.get_val(child.lemma_) != 0:
                        child_value = int(self.get_val(child.lemma_))
                        self.root_deps.append(child_value)
                        self.total_val_attitudes.append(child_value)

                # Entity relations und deren Werte bilden
                if len(self.reversed_entities) != 0 and len(self.root_deps) != 0:

                    if ('ng' in self.subcat_children) == False:
                        self.total_val_relations[self.reversed_entities[0]] = self.get_sum(self.root_deps)
                        self.reversed_entities.pop(0)

                    else:
                        self.rev_sentiments = True
                        self.total_val_relations[self.reversed_entities[0]] = self.rev(self.get_sum(self.root_deps))
                        self.reversed_entities.pop(0)

            except KeyError:
                print('Das Wort ' + str(token) + ' ist nicht in dem Lexikon')

        # Element is used - pop it
        self.tree.popitem(True)  # Pop first element

        # 1. Set attitudes
        # Attention - the self.total_val_attitudes list is REVERSED*
        # * - parsed from inside out (to get the value from daughter nods to the root
        # 2. Check here whether the sentence is negated or not
        value = self.get_sum(self.total_val_attitudes)
        if ('ng' in self.subcat_children) == False:
            self.attitudes[self.sb[0]] = value
        else:
            self.rev_sentiments = True
            self.attitudes[self.sb[0]] = self.rev(value)

        # Nachdem alle Werte extrahiert wurden, werden die Relations berechnet
        self.set_relations(self.total_val_relations)
        # print(self.total_val_relations)

    ##############################################
    ######### Methode für die Relations ##########
    ########### zwischen den Entitäten ###########
    ##############################################

    # 1. Die Relations werden nur dann gesetzt, wenn
    #    a) die Liste von Entitäten definiert von SpaCy
    #       nicht leer ist
    #    b) die Liste mindestens 1 Element enthält und es
    #       mit dem Subjekt des Satzes nicht identisch ist
    #    c) Bei mehr als zwei Entitäten in der Liste werden
    #       mit der Funktion self.pairs() Paare von Entitäten gebildet
    #
    # ! Nicht alle Entitäten werden von SpaCy korrekt erfasst

    def set_relations(self, entities):

        ents = list(entities.keys())

        # Wenn nur 1 Entität im Satz und die Entität ist nicht Subjekt des Satzes
        if len(entities) == 1 and str(list(entities.keys())[0]) != self.sb[0][1]:
            total_val = self.get_sum(list(entities.values()))
            # print(list(entities.values()))
            self.relations.add((self.sb[0], total_val, ents[0]))

        elif len(entities) == 2:
            total_val = self.get_sum(list(entities.values()))
            # print(total_val)
            self.relations.add((ents[1], total_val, ents[0]))

        elif len(entities) > 2:
            # print(self.ent_pairs)
            for pair in self.ent_pairs:
                values = self.calc_two_values(entities[pair[0]], entities[pair[1]])
                self.relations.add((pair[0], values, pair[1]))

    ##############################################
    ######### Methode für den Zugriff ############
    ######### auf die Werte im Lexikon ###########
    ##############################################

    def get_val(self, arg):

        if arg in self.lexicon:
            return int(self.lexicon[arg])

        return 0

    ##############################################
    ######### Methode für die Berechnung #########
    ############## von zwei Werten ###############
    ##############################################

    def calc_two_values(self, v1, v2):
        if v1 == -1 and v2 == 1:
            return v1
        if v1 == 1 and v2 == -1:
            return v2
        if v1 == 0 and v2 == -1:
            return v2
        if v1 == 0 and v2 == 1:
            return v2
        if v1 == -1 and v2 == 0:
            return v1
        if v1 == -1 and v2 == -1:
            return 1
        if v1 == 1 and v2 == 1:
            return 1

        return 0

    ##############################################
    ######### Methode für die Berechnung #########
    ###########  von Attitude-Wert ###############
    ##############################################

    # Attitude-Wert ist von der Liste von allen Werten (self.total_val_attitude),
    # die die Funkton analyze gefunden hat, berechnet
    # Hier richtet sich die Methode auch nach der Länge der Liste

    def get_sum(self, piece):
        if len(piece) == 0:
            sum = 0
            return sum
        elif len(piece) == 1:
            sum = piece[0]
            return sum
        elif len(piece) == 2:
            sum = self.calc_two_values(piece[0], piece[1])
            return sum
        else:
            value = self.calc_two_values(piece[0], piece[1])
            piece[:2] = [value]
            sum = self.get_sum(piece)
            return sum

    ##############################################
    #### Methode für das Umkehren der Werte #####
    ########  wenn Negation vorhanden ist ########
    ##############################################

    def rev(self, val):
        if self.rev_sentiments == True:

            if val == -1:
                return 1
            elif val == 1:
                return -1

        return 0

    ##############################################
    #### Methode, die Paare von Entitäten  #######
    ########  für die Relationen bildet ##########
    ##############################################

    def pairs(self, entities):
        pairs = []

        if len(entities) > 1:
            for j in range(1, len(entities)):
                pair = (entities[j], entities[j - 1])
                pairs.append(pair)

        return pairs

################################################
#### Klasse zur Aufbau der Satzstrukur #########
################################################

class Tree():

    def __init__(self, sent):

        self.tree = self.build_tree(sent)
        self.ents = sent.ents
        self.sent_tags = []
        self.sent_deps = []
        # print(self.sent_tags)
        # print(self.sent_deps)

    ################################################
    #### Methode zur Aufbau von Baumstruktur #######
    ################################################

    # 1. Die nicht relevanten Tags und Dependenzen werden durch die
    #    nonsentiment Listen ignoriert
    # 2. Der Baum wird in Form von OrderdDict gespeichert, da die Reihenfolge
    #    von Schlüssel wichtig ist - die müssen dann reversed analysiert werden
    #    (bottom-up)

    # To get the verb prefixes correctly !!!
    #        for item in children:
    #                 if item.dep_ == 'svp':
    #                     token = str(item) + token
    #                     print(token)
    #             else:
    #                 token = word[0].lemma_

    def build_tree(self, sent):
        sentiment_tags = ['ROOT', 'VVFIN', 'VVPP', 'VVINF']
        nonsentiment_deps = ['punct', 'cp']
        nonsentiment_tags = ['PTKZU', '$,','$.','KOUS','ART']

        tree = OrderedDict()
        for token in sent:
            if token.dep_ not in nonsentiment_deps:
                if token.dep_ == 'ROOT' or token.tag_ in sentiment_tags:
                    parse = token.text
                    children = [child for child in token.children if
                                child.tag_ not in nonsentiment_tags and child.tag_ not in sentiment_tags]
                    if len(children) != 0:
                        tree[parse] = children

        return tree

    def check_noun_phrase(self):

        self.pp = []
        self.np = []
        self.sb = []

        for item in sent:
            self.sent_tags.append(item.tag_)
            self.sent_deps.append(item.dep_)

        for key, value in self.tree.items():

            for item in value:
                if item.dep_ == 'sb':
                    self.sb.append((next(item.children, '-'), item.text))
                    value += item.children

                elif item.tag_ == 'NN' and len(list(item.children)) != 0 and 'APPR' not in self.sent_tags:
                    self.np = (item.text, item.children)
                    value += item.children

                # von der Niederlage der Engländer
                elif item.tag_ == 'APPR':
                    self.pp = list(item.subtree)
                    print(self.pp)
                    value += self.pp[1:]

                elif item.tag_ == 'ADJD':
                    self.pp.append(str(item))

        return self.tree


if __name__ == "__main__":

    start_time = time.time()

    ############################
    ########## Sätze ###########
    ############################

    # Welche syntaktische Strukturen von dem Programm unerstützt werden:
    # sent = u'Er profitiert'
    # sent = u'Er hofft'
    # sent = u'Er gewinnt'
    # sent = u'Er betrog die Menschen'
    # sent = u'Peter ist böse'
    # sent = u'Peter ist freundlich'
    # sent = u'Peter sorgt für die schlechte Stimmung'
    # sent = u'Peter sorgt für den Sieg'
    # sent = u'Peter profitiert von der Krise'
    # sent = u'Er profitiert davon'
    # sent = u'Er profitiert davon nicht'
    # sent = u'Kroaten helfen den Deutschen'
    # sent = u'Kroaten helfen den Deutschen nicht'
    # sent = u'Peter hilft Marie'
    # sent = u'Er freut sich auf ihren Misserfolg'
    # sent = u'Er freut sich auf ihren Erfolg'
    # sent = u'Peter profitiert davon, dass die Krise herrscht'
    # sent = u'Peter hilft Marie dabei nicht, die schlechte Aufgabe zu machen'
    # sent = u'Peter hilft Marie dabei, die gute Aufgabe zu machen'
    # sent = u'Der Stürmer sorgt dafür, dass die Mannschaft gewinnt'
    # sent = u'Die Engländer profitieren davon, dass die Kroaten verlieren'
    # sent = u'Die Kroaten profitieren davon, dass die Engländer verhindern, dass die Franzosen gewinnen.'
    # sent = u'Die Kroaten profitieren davon nicht, dass die Engländer verhindern, dass die Franzosen gewinnen.'
    # sent = u'Die UN kritisiert , dass Ungarn die EU dabei nicht unterstützt , den Flüchtlingen zu helfen .'
    # sent = u'Die EU sorgt nicht dafür , dass Griechenland die Genesung gelingt .'
    # sent = u'Griechenland befürchtet zu scheitern .'
    # sent = u'Stefan kritisiert , dass Marie bedauert , den Peter belogen zu haben .'
    # sent = u'Peter wirft vor, dass Marie lügt'
    sent = u"Deutschland und Frankreich werfen Grossbritanien vor,dass es die Beziehungen zu den USA nicht bevorzugen will."

    ####################################
    ########## Instanz der Klasse ######
    ############## erzeugen ############
    ####################################

    nlp = spacy.load('de_core_news_sm')
    sent = nlp(sent)
    root = [token for token in sent if token.dep_ == 'ROOT'][0]

    doc = Role_Modeling(root)
    struct = Tree(sent)
    tree = struct.build_tree(sent)

    print("-%-%-%-%-%-%-%-%-%-%-%- OUTPUT -%-%-%-%-%-%-%-%-%-%-%-")
    print("Sentence: " + str(sent))
    print("Attitudes: " + str(doc.attitudes))
    print("Relations: " + str(doc.relations))

    print("Runtime: " + "--- %s seconds ---" % (time.time() - start_time))