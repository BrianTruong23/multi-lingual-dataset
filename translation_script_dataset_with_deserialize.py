import base64
from io import BytesIO
from PIL import Image
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd
from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText
import torch
import gc
import tqdm
import re
import time
import os
from datasets import load_dataset
import json

# Function to translate a batch of texts using the SeamlessM4T model
def translate_batch(texts, processor, model, device, tgt_lang):

    if texts == '':
      return ''

    text_inputs = processor(
        text=texts,
        src_lang="eng",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            **text_inputs, tgt_lang=tgt_lang, max_new_tokens=256
        )
    translated_texts = [
        processor.decode(ids, skip_special_tokens=True) for ids in generated_ids
    ]

    # GPU Garbage collection
    del text_inputs
    torch.cuda.empty_cache()
    gc.collect()

    return translated_texts

# Function to translate the 'value' field in each dictionary of the 'conversations' list
def translate_column(column_original_list, processor, model, device, tgt_lang):

    values = column_original_list
    print("Values to Translate: ", len(values))

    # Translate the batch of values
    translated_values = []
    batch_size = 8  # Set batch size
    print("Using batch size: ", batch_size)
    for i in tqdm.tqdm(range(0, len(values), batch_size)):
        batch_values = values[i : i + batch_size]
        translated_batch = translate_batch(
            batch_values, processor, model, device, tgt_lang
        )
        translated_values.extend(translated_batch)

        # Clear cache after each batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print("Successfully Translated: ", len(translated_values))

    return translated_values

def image_to_base64(img):
    if img is None:
        return None
    if isinstance(img, str):  # If it's already a string (perhaps a file path), return it as is
        return img
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def serialize_complex_columns(df):
    for column in df.columns:
        if 'image' in column.lower():
            df[column] = df[column].apply(image_to_base64)
        elif df[column].dtype == 'object':
            df[column] = df[column].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
    return df

def base64_to_image(b64_string):
    if b64_string is None:
        return None
    if not isinstance(b64_string, str):
        return b64_string  # Return as is if it's not a string
    try:
        # Try decoding the base64 string and converting to an image
        img_data = base64.b64decode(b64_string)
        print("IMAGE DATA conversion")
        return Image.open(BytesIO(img_data))
    except Exception as e:
        print(f"Failed to convert base64 to image: {e}")
        return b64_string  # Return as is if conversion fails



def safe_json_loads(x):
    if not isinstance(x, str):
        return x
    if not (x.startswith('{') or x.startswith('[')):
        return x
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        return x

def deserialize_complex_columns(df):
    for column in df.columns:
        if 'image' in column.lower():
            print('Applying Serialize Image Column')
            df[column] = df[column].apply(base64_to_image)

        elif df[column].dtype == 'object':
            df[column] = df[column].apply(safe_json_loads)
    return df

def data_translate_column(processor, model, device, tgt_language):
    domains = [
        "Accounting",
        # Add more domains if needed
    ]

    for domain in domains:
        dataset = load_dataset("MMMU/MMMU", domain)

        # Convert to Pandas DataFrame
        df_dev = pd.DataFrame(dataset['dev'])
        df_test = pd.DataFrame(dataset['test'])
        df_validation = pd.DataFrame(dataset['validation'])

        # Serialize complex columns
        df_dev = serialize_complex_columns(df_dev)
        # df_test = serialize_complex_columns(df_test)
        # df_validation = serialize_complex_columns(df_validation)

        # Translate the columns of df_dev
        columns = ['question', 'options', 'explanation']
        for column in columns:
            df_dev[column] = translate_column(df_dev[column].tolist(), processor, model, device, tgt_language)
            # df_test[column] = translate_column(df_test[column].tolist(), processor, model, device, tgt_language)
            # df_validation[column] = translate_column(df_validation[column].tolist(), processor, model, device, tgt_language)

        # Deserialize complex columns
        df_dev = deserialize_complex_columns(df_dev)
        # df_test = deserialize_complex_columns(df_test)
        # df_validation = deserialize_complex_columns(df_validation)

        # Convert DataFrames to Datasets
        dataset_dev = Dataset.from_pandas(df_dev)
        # dataset_validation = Dataset.from_pandas(df_validation)
        # dataset_test = Dataset.from_pandas(df_test)

        # Create a DatasetDict
        dataset_dict = DatasetDict({
            'dev': dataset_dev,
            # 'validation': dataset_validation,
            # 'test': dataset_test
        })

        # Save the dataset
        dataset_dict.save_to_disk(f'{tgt_language}/MMMU/{domain}')


# def data_translate_column(processor, model, device, tgt_language):
#     domains = [
#         # 2 hours
#         "Accounting",
#         # "Agriculture",
#         # "Architecture_and_Engineering",
#         # "Art",
#         # "Art_Theory",
#         # "Basic_Medical_Science",
#         # "Biology",
#         # "Chemistry",
#         # "Clinical_Medicine",
#         # "Computer_Science",
#         # "Design",
#         # "Diagnostics_and_Laboratory_Medicine",
#         # "Economics",
#         # "Electronics",
#         # "Energy_and_Power",
#         # "Finance",
#         # "Geography",
#         # "History",
#         # "Literature",
#         # "Manage",
#         # "Marketing",
#         # "Materials",
#         # "Math",
#         # "Mechanical_Engineering",
#         # "Music",
#         # "Pharmacy",
#         # "Physics",
#         # "Psychology",
#         # "Public_Health",
#         # "Sociology"
#     ]

#     for domain in domains:
#         dataset = load_dataset("MMMU/MMMU", domain)

#         # Convert to Pandas dataframe
#         df_dev = pd.DataFrame(dataset['dev'])
#         df_test = pd.DataFrame(dataset['test'])
#         df_validation = pd.DataFrame(dataset['validation'])

#         # Serialize complex columns
#         df_dev = serialize_complex_columns(df_dev)
#         df_test = serialize_complex_columns(df_test)
#         df_validation = serialize_complex_columns(df_validation)

#         # Translate the columns of df_dev 
#         columns = ['question', 'options', 'explanation']
#         for column in columns:
#           df_dev[column] = translate_column(df_dev[column].tolist(), processor, model, device, tgt_language)
#           df_test[column] = translate_column(df_test[column].tolist(), processor, model, device, tgt_language)
#           df_validation[column] = translate_column(df_validation[column].tolist(), processor, model, device, tgt_language)

#         # Deseiralize complex columns:
#         df_dev = des
#         # Convert DataFrames to Datasets
#         dataset_dev = Dataset.from_pandas(df_dev)
#         dataset_validation = Dataset.from_pandas(df_validation)
#         dataset_test = Dataset.from_pandas(df_test)

#         # Create a DatasetDict
#         dataset_dict = DatasetDict({
#             'dev': dataset_dev,
#             'validation': dataset_validation,
#             'test': dataset_test
#         })

#         # Save the dataset
#         dataset_dict.save_to_disk(f'{tgt_language}/MMMU/{domain}')

  
def main():
    start_time = time.time()
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

     # Clear GPU memory at the start
    torch.cuda.empty_cache()
    gc.collect()

    # Initialize the SeamlessM4T processor and model
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForTextToText.from_pretrained(
        "facebook/seamless-m4t-v2-large"
    ).to(device)

    print("\nSTARTING TRANSLATION\n")

    # Starting translation
    tgt_language = 'vie'

    data_translate_column(processor, model, device, tgt_language)

    print("\nTRANSLATION COMPLETE\n")
    end_time = time.time()  # End timing
    total_time = end_time - start_time
    print(f"Total time taken for translation: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()