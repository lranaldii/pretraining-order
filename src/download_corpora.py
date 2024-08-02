import os
from datasets import load_dataset

# Create datasets directory
os.makedirs('datasets', exist_ok=True)
os.chdir('datasets')


def clone_dataset(repo_name, subset=None, columns=None, language=None, date=None):
    if language and date:
        dataset = load_dataset(repo_name, language=language, date=date, split=subset)
    else:
        dataset = load_dataset(repo_name, split=subset)  
    if columns:
        dataset = dataset.select_columns(columns)
    return dataset

# Clone and process datasets


mc4_it_clean = clone_dataset('gsarti/clean_mc4_it', subset='tiny')
mc4_en_clean = clone_dataset('liweili/c4_200m', columns=['output'])


wikipedia_it = clone_dataset('legacy-datasets/wikipedia', language='it', date='20220120')
wikipedia_en = clone_dataset('legacy-datasets/wikipedia', language='en', date='20220120')

culturaX = clone_dataset('uonlp/CulturaX', subset='multilingual')
culturaX_it = culturaX.filter(lambda x: x['language'] == 'it')
culturaX_en = culturaX.filter(lambda x: x['language'] == 'en')


mc4_it_clean.save_to_disk('mc4_it_clean_tiny')
mc4_en_clean.save_to_disk('c4_200m_output')
wikipedia_it.save_to_disk('wikipedia_it_20220120')
wikipedia_en.save_to_disk('wikipedia_en_20220120')
culturaX_it.save_to_disk('culturaX_it')
culturaX_en.save_to_disk('culturaX_en')

print("Download and processing completed.")

