def main():
    input = "BraTS2020_training_data/content"
    output = "Outputs"

    preprocess = Preprocessing(input, output, 64, 95)
    preprocess.svd()


if __name__ == "__main__":
    main()
