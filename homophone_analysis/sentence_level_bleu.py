#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nltk.translate.bleu_score import sentence_bleu

references_raw = ["Zu Hause in New York, bin ich Chef der Entwicklungsabteilung einer gemeinnützigen Organisation namens Robin Hood.",
"Wenn ich nicht die Armut bekämpfe, bekämpfe ich als Gehilfe eines Feuerwehr-Hauptmanns bei einem freiwilligen Löschzug das Feuer.",
"Nun, in unserer Stadt, in der Freiwillige eine hochqualifizierte Berufsfeuerwehr unterstützten, muss man ziemlich früh an der Brandstelle sein, um mitmischen zu können.",
"Ich erinnere mich an mein erstes Feuer."]

candidate_raw = ["In New York bin ich der Chef der Entwicklung für eine gemeinnützige namens Robben .",
"Wenn ich Armut nicht bekämpfe , kämpfe ich gegen Feuer .",
"In unserer Stadt , wo die Freiwilligen einen hochqualifizierten Karrierepfad eingesperrt haben , muss man dem Feuer recht früh ins Spiel kommen und auf jegliches Verhalten kommen .",
"Ich erinnere mich an meinen ersten Feuer ."]

references = [l.split() for l in references_raw]
candidates = [l.split() for l in candidate_raw]

def main():
    for ref, cand in zip(references,candidates):
        print(ref, cand)
        print(sentence_bleu(ref, cand))
        print("\n")

if __name__ == "__main__":
    main()

# reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
# candidate = ['this', 'is', 'a', 'test']
# score = sentence_bleu(reference, candidate)
# print(score)


