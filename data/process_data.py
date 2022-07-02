"""
Preprocessing of Data
Project: Disaster Response Pipeline (Udacity - Data Science Nanodegree)
Sample Script Syntax:
> python process_data.py <path to messages csv file> <path to categories csv file> <path to sqllite  destination db>
Sample Script Execution:
> python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db
Arguments Description:
    1) Path to the CSV file containing messages (e.g. disaster_messages.csv)
    2) Path to the CSV file containing categories (e.g. disaster_categories.csv)
    3) Path to SQLite destination database (e.g. disaster_response_db.db)
"""

# Import all the relevant libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
 
def load_messages_with_categories(messages_filepath, categories_filepath):
    """
    Load Messages Data with Categories Function
    
    Arguments:
        messages_filepath -> Path to the CSV file containing messages
        categories_filepath -> Path to the CSV file containing categories
    Output:
        df -> Combined data containing messages and categories
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(left=messages, right=categories, left_index=True, right_index=True, validate='1:1')
    df.drop(columns='id_y', inplace=True)
    df.rename(columns={'id_x':'id'}, inplace=True)
    return df 

def clean_categories_data(df):
    """
    Clean Categories Data Function
    
    Arguments:
        df -> Combined data containing messages and categories
    Outputs:
        df -> Combined data containing messages and categories with categories cleaned up
    """
    
    # Split the categories
    categories = df['categories'].str.split(pat=';',expand=True)
    
    #Fix the categories columns name
    row = categories.iloc[0,:]
    category_colnames = row.str.extract(r'(\w*)-\d')[0].values
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories.loc[:,column].str.extract(r'\w*-(\d)')[0]
        categories[column] = categories[column].astype(int)
    categories.related.replace(2,1,inplace=True) #assume 2 was supposed to be 1
    categories.drop(columns='child_alone', inplace=True) # single class
    
    # drop original and replace by clean categories
    df.drop(columns='categories', inplace=True)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
    
    return df

def save_data_to_db(df, database_filepath):
    """
    Save Data to SQLite Database Function
    
    Arguments:
        df -> Combined data containing messages and categories with categories cleaned up
        database_filename -> Path to SQLite destination database
    """
    
    engine = create_engine('sqlite:///'+database_filepath)
    df.to_sql('DisasterResponse', engine,if_exists = 'replace', index=False)  

def main():
    """
    Main function which will kick off the data processing functions. There are three primary actions taken by this function:
        1) Load and merge messages dataframe with categories
        2) Clean Categories Data and return df
        3) Save df to SQLite Database
    """
    
    # Print the system arguments
    # print(sys.argv)
    
    # Execute the ETL pipeline if the count of arguments is matching to 4
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filename = sys.argv[1:] # Extract the parameters in relevant variable

        print('Loading messages data from {} ...\nLoading categories data from {} ...'
              .format(messages_filepath, categories_filepath))
        
        df = load_messages_with_categories(messages_filepath, categories_filepath)

        print('Cleaning categories data ...')
        df = clean_categories_data(df)
        
        print('Saving data to SQLite DB : {}'.format(database_filename))
        save_data_to_db(df, database_filename)
        
        print('Cleaned data has been saved to database!')
    
    else: # Print the help message so that user can execute the script with correct parameters
        print("Please provide the arguments correctly: \nSample Script Execution:\n\
> python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db \n\
Arguments Description: \n\
1) Path to the CSV file containing messages (e.g. disaster_messages.csv)\n\
2) Path to the CSV file containing categories (e.g. disaster_categories.csv)\n\
3) Path to SQLite destination database (e.g. disaster_response_db.db)")

if __name__ == '__main__':
    main()