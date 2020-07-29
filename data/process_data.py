import pandas as pd
import sqlite3
import sys

def load_dataframes(mess, cat):
    '''Input: two filepaths to csv files.
    Function reads the csv files, and returns pandas dataframes.
    '''
    messages = pd.read_csv(mess, index_col='id')
    categories = pd.read_csv(cat, index_col='id')

    return messages, categories

def clean_categories(cat):
    '''Input: Pandas dataframe of message categories.

    Function splits the column with all categories into one column per category.

    Columns are renamed to match, and data is cleaned to only consist of 0's
    and 1's.
    '''
    #identify all column names
    cat_names = cat['categories'][2].replace('-1', '').replace('-0', '')
    cat_names = cat_names.split(';')

    #create a dictionary of column names and numbers for use in renaming cols.
    col_mapper = {}
    for x, y in enumerate(cat_names):
        col_mapper[x] = y

    #split columns
    cat = cat['categories'].str.split(
                    ';', len(cat_names), expand=True)

    #rename columns using our dictionary
    cat = cat.rename(col_mapper, axis=1)

    #clean data to only consist of 0's and 1's
    for col in cat.columns:
        cat[col] = cat[col].apply(lambda x: int(x.split('-')[1]))

    #drop column child_alone, as it is all zeros.
    cat = cat.drop(['child_alone'], axis=1)

    #there are some 2's in column 'related'
    #replacing these with 1's
    cat['related'] = cat['related'].apply(lambda x:
                        '1' if x=='2' else x)

    return cat

def merge_datasets(mess, cat):
    '''Input: Original messages df, and cleaned categories df.

    Function merges the two and drops duplicates.
    '''
    merged = mess.merge(cat, left_index=True, right_index=True)

    merged.drop_duplicates(inplace=True)

    return merged

def output_db(df, dbpath):
    '''Input: merged df.

    Function creates a sqlite3 database (CategorisedMessages), and stores the
    merged df as a table called Messages.
    '''
    conn = sqlite3.connect(dbpath)

    df.to_sql('Messages', conn, if_exists='replace')

def pipeline(mess, cat, dbpath):
    '''Input:
        mess: file path to messages dataframe.
        cat: file path to categories dataframe.
        dbpath: file path for the database that will be created.

    Function runs the full ETL pipeline.

    Output is a sqlite3 database with cleaned and merged data.
    '''
    messages, categories = load_dataframes(mess, cat)
    categories = clean_categories(categories)
    merged = merge_datasets(messages, categories)
    output_db(merged, dbpath)

if __name__ == '__main__':
    df_messages = sys.argv[1]
    df_categories = sys.argv[2]
    db_path = sys.argv[3]
    pipeline(df_messages, df_categories, db_path)
