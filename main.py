import pathlib

import numpy as np
import orjson
import tqdm
from scipy import sparse
from sknetwork.ranking import PageRank

data_root = pathlib.Path("~/data/archive").expanduser()


def main():
    # mapping form db id to nodes id
    subjects: dict[int, int] = {}
    persons: dict[int, int] = {}
    characters: dict[int, int] = {}

    nodes = []

    print("loading subject.jsonlines")
    with data_root.joinpath("subject.jsonlines").open(encoding="utf8") as f:
        for line in tqdm.tqdm(f.readlines(), ascii=True):
            s = orjson.loads(line)
            nodes.append(sum(s["score_details"].values()) + 1)
            subjects[s["id"]] = len(nodes) - 1

    print("loading person.jsonlines")
    with data_root.joinpath("person.jsonlines").open(encoding="utf8") as f:
        for line in tqdm.tqdm(f.readlines(), ascii=True):
            s = orjson.loads(line)
            nodes.append(1)
            persons[s["id"]] = len(nodes) - 1

    print("loading character.jsonlines")
    with data_root.joinpath("character.jsonlines").open(encoding="utf8") as f:
        for line in tqdm.tqdm(f.readlines(), ascii=True):
            s = orjson.loads(line)
            nodes.append(1)
            characters[s["id"]] = len(nodes) - 1

    edges = {}

    print("loading subject-relations.jsonlines")
    with data_root.joinpath("subject-relations.jsonlines").open(encoding="utf8") as f:
        for line in tqdm.tqdm(f.readlines(), ascii=True):
            s = orjson.loads(line)
            subject_id = s["subject_id"]
            related_subject_id = s["related_subject_id"]

            if subject_id not in subjects:
                continue
            if related_subject_id not in subjects:
                continue

            key = (subjects[subject_id], subjects[related_subject_id])
            edges[key] = edges.get(key, 0) + 1

    print("loading subject-characters.jsonlines")
    with data_root.joinpath("subject-characters.jsonlines").open(encoding="utf8") as f:
        for line in tqdm.tqdm(f.readlines(), ascii=True):
            s = orjson.loads(line)
            subject_id = s["subject_id"]
            character_id = s["character_id"]

            if subject_id not in subjects:
                continue
            if character_id not in characters:
                continue

            key = (subjects[subject_id], characters[character_id])
            edges[key] = edges.get(key, 0) + 1

            key = (characters[character_id], subjects[subject_id])
            edges[key] = edges.get(key, 0) + 1

    print("loading subject-persons.jsonlines")
    with data_root.joinpath("subject-persons.jsonlines").open(encoding="utf8") as f:
        for line in tqdm.tqdm(f.readlines(), ascii=True):
            s = orjson.loads(line)
            subject_id = s["subject_id"]
            person_id = s["person_id"]

            if subject_id not in subjects:
                continue
            if person_id not in persons:
                continue

            key = (subjects[subject_id], persons[person_id])
            edges[key] = edges.get(key, 0) + 1

            key = (persons[person_id], subjects[subject_id])
            edges[key] = edges.get(key, 0) + 1

    print("building edge matrix")
    mtx = sparse.dok_matrix((len(nodes), len(nodes)), dtype=np.uint32)

    for key, value in edges.items():
        mtx[key[0], key[1]] = value

    # Initialize PageRank algorithm
    pagerank = PageRank()

    # Fit the model
    print("pagerank.fit_predict")
    scores = pagerank.fit_predict(mtx.tocoo(), weights=np.array(nodes))

    print("PageRank scores:", scores)

    subject_scores = {s: scores[i] for s, i in subjects.items()}
    # character_scores = {s: scores[s] for s in characters}
    # person_scores = {s: scores[s] for s in persons}

    for r, score in sorted(subject_scores.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]:
        print("https://bgm.tv/subject/{}".format(r), nodes[subjects[r]], score)


if __name__ == "__main__":
    main()
