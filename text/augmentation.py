import nlpaug.augmenter.word as naw
import pandas as pd
from tqdm import tqdm

PATH = "./data"


def aug(article, garment_group_dict, N):
    data = article[["detail_desc", "garment_group_name", "garment_group_no"]]
    data["label"] = data["garment_group_name"].map(garment_group_dict)

    data = data.drop_duplicates(["detail_desc", "label"]).dropna().reset_index(drop=True)
    aug_data = {"detail_desc": [], "label": []}

    cwe = naw.ContextualWordEmbsAug()  # Contextual Word Embeddings - Word level
    syn = naw.SynonymAug(aug_src="wordnet")  # Synonym

    print("##################### DATA AUG #####################")
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        text, label = row.detail_desc, row.label

        aug_data["detail_desc"].extend(cwe.augment(data=text, n=N))
        aug_data["label"].extend([label for _ in range(N)])
        aug_data["detail_desc"].extend(syn.augment(data=text, n=N))
        aug_data["label"].extend([label for _ in range(N)])

    pd.concat([data, pd.DataFrame(aug_data)], axis=0).to_csv(f"{PATH}/aug_item_{N}.csv", index=False)


if __name__ == "__main__":
    N = 3
    article = pd.read_csv(f"{PATH}/articles.csv")
    garment_group_dict = {k: v for v, k in enumerate((article.garment_group_name).unique())}

    aug()
