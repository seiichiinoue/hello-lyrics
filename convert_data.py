import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("pkshatech/simcse-ja-bert-base-clcmlp")


def split_lyrics(data: pd.DataFrame) -> pd.DataFrame:
    lyrics_split = []
    for i in range(len()):
        lyrics_split.append(data["lyrics"][i].split())
    data["lyrics_split"] = lyrics_split
    return data


def convert(
    data_path: str = "data/country-girls.tsv",
    output_path: str = "log_dir",
    split: bool = True,
) -> None:
    data = pd.read_csv(data_path, delimiter="\t")
    data = data.rename(
        columns={
            "曲名": "title",
            "発売日": "release_date",
            "作詞者": "words",
            "作曲者": "music",
            "歌詞": "lyrics",
        }
    )
    tar_col = "lyrics"
    if split:
        data = split_lyrics(data)
        tar_col = "lyrics_split"
    sentences = sum(data[tar_col].values.tolist(), [])
    indices = np.cumsum([len(tar) for tar in data[tar_col].values])
    embeddings = model.encode(sentences)
    embeddings = np.split(embeddings, indices[:-1])
    with open(f"{output_path}/embeddings/feature_vecs.tsv", "w") as f1, open(
        f"{output_path}/embeddings/metadata.tsv", "w"
    ) as f2:
        f2.write(f"title\twords\tmusic\n")
        for i, embs in enumerate(embeddings):
            for emb in embs:
                f1.write("\t".join(map(str, emb)) + "\n")
                f2.write(
                    f"{data['title'][i]}\t{data['words'][i]}\t{data['music'][i]}\n"
                )


if __name__ == "__main__":
    convert()
