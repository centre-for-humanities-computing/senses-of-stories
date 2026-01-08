This repo was used to process data and explore results for the *Senses of Stories* project.
It is a collaboration between several Canadian and European universities and is led by McGill University

The repo is structured as follows:

```
├── README.md
├── data/
│   └── org_data_processing/
│       ├── README.md              # Description of the columns in the dataset
│       ├── process_data.py        # Script to process raw data files
│       └── data/                  # Raw and processed data files (not pushed)
├── notebooks/
│   ├── checkout_corrs.py          # Exploratory data analysis
│   ├── find_example_sentences.py  # Find example sentences from the dataset
│   └── exploring_results.ipynb    # Explore annotated data
├── src/
│   ├── utils.py                   # Utility functions
│   ├── process_data.py            # Process raw data files
│   └── remove_sexual_content.py   # Filter sexual/profanity content
├── results/
└── figs/                          # Generated figures
```
