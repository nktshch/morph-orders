"""Converts .conllu to .parquet"""

from pathlib import Path
import pandas


def convert_conllu_to_parquet(conllu_path, parquet_path):
    data = []
    with open(conllu_path, 'r', encoding='utf-8') as f:
        sentences = f.read().split('\n\n')
        for i, sentence in enumerate(sentences):
            if sentence == '':
                continue
            sentence_data = {'id': i, 'tokens': [], 'tags': []}
            id_word_tag_triples = sentence.split('\n')
            for id_word_tag in id_word_tag_triples:
                if id_word_tag:
                    if id_word_tag[0] != '#':
                        # 0th field is id
                        # 1st field is token (wordform)
                        # 3rd field is pos (upos)
                        # 5th field is other features
                        word_id, token, pos, feats = [id_word_tag.split('\t')[i] for i in [0, 1, 3, 5]]
                        if '.' not in word_id and '-' not in word_id:
                            tag = f'POS={pos}|{feats}' if feats != '_' else f'POS={pos}'
                            # POS=NOUN|Animacy=Inan|Case=Acc|Gender=Masc|Number=Sing
                            # or
                            # POS=SCONJ
                            sentence_data['tokens'].append(token)
                            sentence_data['tags'].append(tag)
            data.append(sentence_data)
        pandas.DataFrame(data).to_parquet(parquet_path, index=False)


def main():
    # CONVERT TO PARQUET
    p = Path('./data')
    files = [str(x) for x in p.rglob('*.conllu')]
    for f in files:
        convert_conllu_to_parquet(f, f.replace('.conllu', '.parquet'))
        print(f'{f} -> {f.replace(".conllu", ".parquet")}')


if __name__ == '__main__':
    main()
