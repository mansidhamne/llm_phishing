from pathlib import Path
import argparse

from train import train_baselines, train_llms
from infer import make_inference


def getArgs():
    parser = argparse.ArgumentParser(description="Parse arguments from command input.")
    parser.add_argument(
        "-t",
        "--task",
        action="store",
        required=True,
        type=str,
        choices=["train", "infer"],
        help='Enter "train" To train models. Enter "infer" to make an inference. ',
    )
    parser.add_argument(
        "-mt",
        "--model_type",
        action="store",
        type=str,
        default="baseline",
        choices=["LLM", "baseline"],
        help="Choose model type to train.",
    )
    parser.add_argument(
        "-c",
        "--csv_name",
        action="store",
        type=str,
        default="data.csv",
        help="Enter the csv file name for training dataset.",
    )
    parser.add_argument(
        "-l",
        "--label_col",
        action="store",
        type=str,
        default="gen_label",
        help="Enter name of the binary classification column in the csv dataset.",
    )
    parser.add_argument(
        "-n",
        "--text_col",
        action="store",
        type=str,
        default="Messages",
        help="Enter name of the text column in the csv dataset.",
    )
    parser.add_argument(
        "-i",
        "--text_input",
        action="store",
        type=str,
        default="Sample text input for inference.",
        help="Enter message to be tested.",
    )
    return parser.parse_args()


if __name__ == "__main__":

    Path("outputs/csv").mkdir(parents=True, exist_ok=True)
    Path("outputs/scores").mkdir(parents=True, exist_ok=True)
    Path("outputs/model").mkdir(parents=True, exist_ok=True)

    arg = getArgs()

    DATASET_NAME = arg.csv_name
    LABEL_COL = arg.label_col
    TEXT_COL = arg.text_col
    TEXT_INPUT = arg.text_input

    print(TEXT_INPUT)

    if arg.task == "train":
        if arg.model_type == "baseline":
            print("training baseline models ...")
            train_baselines(
                dataset_name=DATASET_NAME,
                label_col=LABEL_COL,
                text_col=TEXT_COL,
            )
        elif arg.model_type == "LLM":
            print("train LLM model")
            train_llms(
                dataset_name=DATASET_NAME,
                label_col=LABEL_COL,
                text_col=TEXT_COL,
            )

    elif arg.task == "infer":
        if arg.model_type == "baseline":
            pred = make_inference(
                user_input=TEXT_INPUT,
                dataset_name=DATASET_NAME,
                label_col=LABEL_COL,
                text_col=TEXT_COL,
            ).best_baseline()
        elif arg.model_type == "LLM":
            pred = make_inference(
                user_input=TEXT_INPUT,
                dataset_name=DATASET_NAME,
                label_col=LABEL_COL,
                text_col=TEXT_COL,
            ).best_llm()
        print(f"Prediction: {pred}")

        if pred == 1:
            print("This message is a spam.")
        else:
            print("This message is not a spam.")